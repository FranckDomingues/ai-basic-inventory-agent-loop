"""FastAPI inventory endpoints backed by a local CSV product store.

The module defines the product data models, CSV persistence operations, and
HTTP routes for listing, creating, updating, and filtering inventory products.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator


app = FastAPI()


@dataclass(frozen=True)
class Product:
    """A product record returned by the inventory API.

    Attributes:
        id: Unique identifier assigned when the product is created.
        name: Product name.
        quantity: Current stock count.
        unit: Unit used to describe the stock count.
    """

    id: str
    name: str
    quantity: int
    unit: str


class ProductCreate(BaseModel):
    """Validated request body for creating an inventory product.

    Names and units must contain at least one non-whitespace character, and
    quantity must be zero or greater.
    """

    name: str
    quantity: int = Field(ge=0)
    unit: str

    @field_validator("name", "unit")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject a name or unit made only of whitespace."""
        if not value.strip():
            raise ValueError("must not be empty or whitespace only")
        return value


class StockDelta(BaseModel):
    """Validated request body for changing a product's stock.

    The integer delta may be positive, negative, or zero; the resulting stock
    quantity is checked by the store before it is persisted.
    """

    delta: int = Field(strict=True)


class CsvProductStore:
    """Read and write inventory product records in a CSV file.

    The default file is ``products.csv`` in the project root. Passing a path
    allows callers and tests to use a separate inventory file.
    """

    fieldnames = ("id", "name", "quantity", "unit")

    def __init__(self, file_path: Path | str | None = None) -> None:
        """Configure the CSV path, using the project data file by default.

        Args:
            file_path: Optional path to the CSV file used by this store.
        """
        self.file_path = (
            Path(file_path)
            if file_path is not None
            else Path(__file__).resolve().parents[1] / "products.csv"
        )

    def create(self, name: str, quantity: int, unit: str) -> Product:
        """Assign an identifier and append a new product to the CSV file.

        Args:
            name: Product name.
            quantity: Initial non-negative stock quantity.
            unit: Unit associated with the quantity.

        Returns:
            The product record that was written.
        """
        product = Product(
            id=str(uuid4()),
            name=name,
            quantity=quantity,
            unit=unit,
        )
        write_header = not self.file_path.exists() or self.file_path.stat().st_size == 0

        with self.file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            # Initialize the CSV schema once, while preserving existing rows.
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
        """Load all products from disk, or return an empty list if absent.

        Returns:
            Product records in CSV order. This read does not create or modify
            the data file.
        """
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
        """Find a product by ID.

        Args:
            product_id: Identifier to search for.

        Returns:
            The matching product, or ``None`` when no record has that ID.
        """
        return next(
            (product for product in self.list_all() if product.id == product_id),
            None,
        )

    def update_quantity(self, product_id: str, delta: int) -> Product | None:
        """Apply a signed stock adjustment and save the inventory.

        Args:
            product_id: Identifier of the product to update.
            delta: Signed integer adjustment to its current quantity.

        Returns:
            The updated product, or ``None`` if the ID is not present.

        Raises:
            ValueError: If applying the delta would make stock negative.
        """
        products = self.list_all()
        product_index = next(
            (index for index, product in enumerate(products) if product.id == product_id),
            None,
        )
        if product_index is None:
            return None

        product = products[product_index]
        updated_quantity = product.quantity + delta
        if updated_quantity < 0:
            raise ValueError("Stock quantity cannot be negative")

        updated_product = Product(
            id=product.id,
            name=product.name,
            quantity=updated_quantity,
            unit=product.unit,
        )
        products[product_index] = updated_product

        # Rewrite the complete CSV so unrelated products remain unchanged.
        with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(
                {
                    "id": item.id,
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit": item.unit,
                }
                for item in products
            )

        return updated_product


product_store = CsvProductStore()


def get_product_store() -> CsvProductStore:
    """Provide the shared CSV store used by inventory route dependencies."""
    return product_store


@app.get("/inventory", response_model=list[Product])
def get_inventory(
    store: CsvProductStore = Depends(get_product_store),
) -> list[Product]:
    """List all stored products.

    Returns:
        HTTP 200 with a JSON array, or an empty array when inventory is empty.
    """
    return store.list_all()


@app.post("/inventory", response_model=Product, status_code=201)
def create_inventory_product(
    product: ProductCreate,
    store: CsvProductStore = Depends(get_product_store),
) -> Product:
    """Create and persist a product from the validated request body.

    Returns:
        HTTP 201 with the created product, including its generated ID.

    Raises:
        HTTP 422: If the name or unit is blank, quantity is negative, or the
            request body does not match the product schema.
    """
    return store.create(product.name, product.quantity, product.unit)


@app.get("/inventory/alerts", response_model=list[Product])
def get_inventory_alerts(
    threshold: int = Query(default=10, ge=0),
    store: CsvProductStore = Depends(get_product_store),
) -> list[Product]:
    """List products with stock strictly below the threshold.

    Args:
        threshold: Non-negative stock boundary; defaults to 10. Products at
            exactly this quantity are not included.

    Returns:
        HTTP 200 with matching products, or an empty array if none match.

    Raises:
        HTTP 422: If the threshold is not a valid non-negative integer.
    """
    return [product for product in store.list_all() if product.quantity < threshold]


@app.patch("/inventory/{product_id}", response_model=Product)
def update_inventory_product(
    product_id: str,
    stock_delta: StockDelta,
    store: CsvProductStore = Depends(get_product_store),
) -> Product:
    """Apply the request's signed stock delta to an existing product.

    Returns:
        HTTP 200 with the updated product.

    Raises:
        HTTP 404: If ``product_id`` does not identify a stored product.
        HTTP 409: If the update would make the quantity negative.
        HTTP 422: If the request body does not contain a valid integer delta.
    """
    try:
        updated_product = store.update_quantity(product_id, stock_delta.delta)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    if updated_product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return updated_product
