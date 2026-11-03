"""FastAPI application base and CSV-backed product persistence."""

import csv
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field, field_validator


app = FastAPI()


@dataclass(frozen=True)
class Product:
    """A product stored in the inventory."""

    id: str
    name: str
    quantity: int
    unit: str


class ProductCreate(BaseModel):
    """Input fields required to register a product."""

    name: str
    quantity: int = Field(ge=0)
    unit: str

    @field_validator("name", "unit")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace only")
        return value


class CsvProductStore:
    """Persist and retrieve products using a CSV file."""

    fieldnames = ("id", "name", "quantity", "unit")

    def __init__(self, file_path: Path | str | None = None) -> None:
        self.file_path = (
            Path(file_path)
            if file_path is not None
            else Path(__file__).resolve().parents[1] / "products.csv"
        )

    def create(self, name: str, quantity: int, unit: str) -> Product:
        """Create a product with a unique identifier and append it to CSV."""
        product = Product(
            id=str(uuid4()),
            name=name,
            quantity=quantity,
            unit=unit,
        )
        write_header = not self.file_path.exists() or self.file_path.stat().st_size == 0

        with self.file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "id": product.id,
                    "name": product.name,
                    "quantity": product.quantity,
                    "unit": product.unit,
                }
            )

        return product

    def list_all(self) -> list[Product]:
        """Return every product currently stored in the CSV file."""
        if not self.file_path.exists():
            return []

        with self.file_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            return [
                Product(
                    id=row["id"],
                    name=row["name"],
                    quantity=int(row["quantity"]),
                    unit=row["unit"],
                )
                for row in reader
            ]

    def get(self, product_id: str) -> Product | None:
        """Return a product by identifier, or None if it is not present."""
        return next(
            (product for product in self.list_all() if product.id == product_id),
            None,
        )


product_store = CsvProductStore()


def get_product_store() -> CsvProductStore:
    """Provide the CSV-backed product store to API routes."""
    return product_store


@app.get("/inventory", response_model=list[Product])
def get_inventory(
    store: CsvProductStore = Depends(get_product_store),
) -> list[Product]:
    """Return all products in the inventory without modifying the CSV file."""
    return store.list_all()


@app.post("/inventory", response_model=Product, status_code=201)
def create_inventory_product(
    product: ProductCreate,
    store: CsvProductStore = Depends(get_product_store),
) -> Product:
    """Create a product and persist it in the inventory CSV file."""
    return store.create(product.name, product.quantity, product.unit)
