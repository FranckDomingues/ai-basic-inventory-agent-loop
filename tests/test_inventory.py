"""Basic tests for product CSV persistence."""

import csv

import pytest
from fastapi.testclient import TestClient

from api.app import CsvProductStore, Product, app, get_product_store


@pytest.fixture
def client(tmp_path):
    store = CsvProductStore(tmp_path / "products.csv")
    app.dependency_overrides[get_product_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client, store
    app.dependency_overrides.clear()


def test_fastapi_app_is_importable() -> None:
    assert app.title == "FastAPI"


def test_create_and_read_product(tmp_path) -> None:
    store = CsvProductStore(tmp_path / "products.csv")

    product = store.create("Apples", 12, "kg")

    assert product == Product(
        id=product.id,
        name="Apples",
        quantity=12,
        unit="kg",
    )
    assert store.get(product.id) == product
    assert store.list_all() == [product]


def test_product_persists_in_csv(tmp_path) -> None:
    file_path = tmp_path / "products.csv"
    store = CsvProductStore(file_path)
    product = store.create("Rice", 5, "bags")

    restored_product = CsvProductStore(file_path).get(product.id)

    assert restored_product == product
    with file_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows == [
        {
            "id": product.id,
            "name": "Rice",
            "quantity": "5",
            "unit": "bags",
        }
    ]


def test_get_inventory_returns_empty_list_without_creating_csv(client) -> None:
    test_client, store = client

    response = test_client.get("/inventory")

    assert response.status_code == 200
    assert response.json() == []
    assert not store.file_path.exists()


def test_get_inventory_returns_all_products_without_mutating_csv(client) -> None:
    test_client, store = client
    first = store.create("Apples", 12, "kg")
    second = store.create("Rice", 5, "bags")
    original_contents = store.file_path.read_bytes()

    response = test_client.get("/inventory")

    assert response.status_code == 200
    assert response.json() == [
        {"id": first.id, "name": "Apples", "quantity": 12, "unit": "kg"},
        {"id": second.id, "name": "Rice", "quantity": 5, "unit": "bags"},
    ]
    assert store.file_path.read_bytes() == original_contents


def test_post_inventory_creates_product_and_persists_csv(client) -> None:
    test_client, store = client
    payload = {"name": "Coffee", "quantity": 3, "unit": "bags"}

    response = test_client.post("/inventory", json=payload)

    assert response.status_code == 201
    created_product = response.json()
    assert created_product["name"] == payload["name"]
    assert created_product["quantity"] == payload["quantity"]
    assert created_product["unit"] == payload["unit"]
    assert created_product["id"]
    assert CsvProductStore(store.file_path).get(created_product["id"]) == Product(
        **created_product
    )


def test_post_inventory_generates_unique_ids(client) -> None:
    test_client, _ = client
    payload = {"name": "Beans", "quantity": 2, "unit": "kg"}

    first_response = test_client.post("/inventory", json=payload)
    second_response = test_client.post("/inventory", json=payload)

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["id"] != second_response.json()["id"]


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "quantity": 1, "unit": "kg"},
        {"name": "   ", "quantity": 1, "unit": "kg"},
        {"name": "Beans", "quantity": 1, "unit": ""},
        {"name": "Beans", "quantity": 1, "unit": "  "},
        {"name": "Beans", "quantity": -1, "unit": "kg"},
    ],
)
def test_post_inventory_rejects_invalid_data(client, payload) -> None:
    test_client, _ = client

    response = test_client.post("/inventory", json=payload)

    assert response.status_code == 422


def test_patch_inventory_increases_stock(client) -> None:
    test_client, store = client
    product = store.create("Apples", 5, "kg")

    response = test_client.patch(
        f"/inventory/{product.id}", json={"delta": 3}
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": product.id,
        "name": "Apples",
        "quantity": 8,
        "unit": "kg",
    }


def test_patch_inventory_decreases_stock(client) -> None:
    test_client, store = client
    product = store.create("Apples", 5, "kg")

    response = test_client.patch(
        f"/inventory/{product.id}", json={"delta": -2}
    )

    assert response.status_code == 200
    assert response.json()["quantity"] == 3


def test_patch_inventory_zero_delta_keeps_stock(client) -> None:
    test_client, store = client
    product = store.create("Apples", 5, "kg")

    response = test_client.patch(
        f"/inventory/{product.id}", json={"delta": 0}
    )

    assert response.status_code == 200
    assert response.json()["quantity"] == 5


def test_patch_inventory_returns_404_for_unknown_product(client) -> None:
    test_client, _ = client

    response = test_client.patch("/inventory/unknown-id", json={"delta": 1})

    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found"


def test_patch_inventory_rejects_insufficient_stock_without_mutation(client) -> None:
    test_client, store = client
    product = store.create("Apples", 2, "kg")
    original_contents = store.file_path.read_bytes()

    response = test_client.patch(
        f"/inventory/{product.id}", json={"delta": -3}
    )

    assert response.status_code == 409
    assert "cannot be negative" in response.json()["detail"]
    assert store.file_path.read_bytes() == original_contents


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"delta": "not-a-number"},
        {"delta": 1.5},
        {"delta": True},
        {"delta": None},
    ],
)
def test_patch_inventory_rejects_invalid_delta(client, payload) -> None:
    test_client, store = client
    product = store.create("Apples", 2, "kg")

    response = test_client.patch(f"/inventory/{product.id}", json=payload)

    assert response.status_code == 422


def test_patch_inventory_quantity_persists_and_get_returns_it(client) -> None:
    test_client, store = client
    product = store.create("Apples", 5, "kg")

    patch_response = test_client.patch(
        f"/inventory/{product.id}", json={"delta": -2}
    )
    restored_store = CsvProductStore(store.file_path)
    get_response = test_client.get("/inventory")

    assert patch_response.status_code == 200
    assert restored_store.get(product.id).quantity == 3
    assert get_response.json() == [
        {"id": product.id, "name": "Apples", "quantity": 3, "unit": "kg"}
    ]


def test_patch_inventory_preserves_unrelated_products(client) -> None:
    test_client, store = client
    target = store.create("Apples", 5, "kg")
    unrelated = store.create("Rice", 10, "bags")

    response = test_client.patch(
        f"/inventory/{target.id}", json={"delta": 2}
    )

    assert response.status_code == 200
    assert CsvProductStore(store.file_path).list_all() == [
        Product(target.id, "Apples", 7, "kg"),
        unrelated,
    ]


def test_get_inventory_alerts_uses_default_threshold_of_ten(client) -> None:
    test_client, store = client
    below = store.create("Apples", 9, "kg")
    at_threshold = store.create("Rice", 10, "bags")
    above = store.create("Coffee", 11, "bags")

    response = test_client.get("/inventory/alerts")

    assert response.status_code == 200
    assert response.json() == [
        {"id": below.id, "name": "Apples", "quantity": 9, "unit": "kg"}
    ]
    assert at_threshold.id not in {item["id"] for item in response.json()}
    assert above.id not in {item["id"] for item in response.json()}


def test_get_inventory_alerts_accepts_custom_threshold(client) -> None:
    test_client, store = client
    low = store.create("Apples", 4, "kg")
    middle = store.create("Rice", 7, "bags")
    high = store.create("Coffee", 10, "bags")

    response = test_client.get("/inventory/alerts?threshold=8")

    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [low.id, middle.id]
    assert high.id not in {item["id"] for item in response.json()}


def test_get_inventory_alerts_returns_empty_for_empty_inventory(client) -> None:
    test_client, _ = client

    response = test_client.get("/inventory/alerts")

    assert response.status_code == 200
    assert response.json() == []


def test_get_inventory_alerts_returns_empty_when_nothing_is_below_threshold(
    client,
) -> None:
    test_client, store = client
    store.create("Rice", 10, "bags")
    store.create("Coffee", 15, "bags")

    response = test_client.get("/inventory/alerts?threshold=10")

    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.parametrize("threshold", ["invalid", "-1", "1.5"])
def test_get_inventory_alerts_rejects_invalid_threshold(client, threshold) -> None:
    test_client, _ = client

    response = test_client.get(f"/inventory/alerts?threshold={threshold}")

    assert response.status_code == 422


def test_get_inventory_alerts_does_not_mutate_csv(client) -> None:
    test_client, store = client
    store.create("Apples", 2, "kg")
    store.create("Rice", 10, "bags")
    original_contents = store.file_path.read_bytes()

    response = test_client.get("/inventory/alerts")

    assert response.status_code == 200
    assert store.file_path.read_bytes() == original_contents
