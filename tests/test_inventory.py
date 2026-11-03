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
