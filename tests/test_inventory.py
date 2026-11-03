"""Basic tests for product CSV persistence."""

import csv

from api.app import CsvProductStore, Product, app


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
