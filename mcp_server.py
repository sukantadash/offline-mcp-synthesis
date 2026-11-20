"""Order Ops sample MCP server.

This process is a reference for the tool schema and the catalog in DESIGN.md.
offline-mcp-synthesis does not start it, connect to it, or call its tools.
The teacher writes simulated tool results from the schema and the sample conversations.

stateful_env.py does start it, once per scenario, with ORDER_OPS_STATE pointing
at a JSON state file. The server then loads its orders and inventory from that
file and writes the updated state back after every successful shipment.

Start, if you want to inspect the tools yourself:

    python mcp_server.py

Streamable HTTP is then available at http://127.0.0.1:8008/mcp.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Optional

from fastmcp import FastMCP

mcp = FastMCP(
    name="Order Ops",
    instructions=(
        "Order and inventory tools for an integration runtime. "
        "Use these to look up orders, check stock, and create shipments."
    ),
)

_ORDERS = {
    "ORD-1001": {
        "id": "ORD-1001",
        "customer": "Acme Retail",
        "sku": "SKU-LAP-14",
        "qty": 2,
        "status": "paid",
        "ship_to": "Austin, TX",
    },
    "ORD-1002": {
        "id": "ORD-1002",
        "customer": "Northwind",
        "sku": "SKU-MON-27",
        "qty": 5,
        "status": "paid",
        "ship_to": "Seattle, WA",
    },
    "ORD-1003": {
        "id": "ORD-1003",
        "customer": "Contoso",
        "sku": "SKU-KB-MECH",
        "qty": 10,
        "status": "pending",
        "ship_to": "Boston, MA",
    },
}

_INVENTORY = {
    "SKU-LAP-14": {"sku": "SKU-LAP-14", "name": "14in Ultrabook", "on_hand": 12, "warehouse": "DFW"},
    "SKU-MON-27": {"sku": "SKU-MON-27", "name": "27in Monitor", "on_hand": 3, "warehouse": "SEA"},
    "SKU-KB-MECH": {"sku": "SKU-KB-MECH", "name": "Mechanical Keyboard", "on_hand": 40, "warehouse": "BOS"},
}

ORDERS: dict[str, dict] = {}
INVENTORY: dict[str, dict] = {}
SHIPMENTS: list[dict] = []


def reset_catalog() -> None:
    """Restore the sample orders, inventory, and shipment log."""
    ORDERS.clear()
    ORDERS.update(copy.deepcopy(_ORDERS))
    INVENTORY.clear()
    INVENTORY.update(copy.deepcopy(_INVENTORY))
    SHIPMENTS.clear()


reset_catalog()

# Optional per-scenario state file: {"orders": {...}, "inventory": {...}, "shipments": [...]}
STATE_FILE = os.environ.get("ORDER_OPS_STATE")


def snapshot() -> dict:
    """Return a copy of the current orders, inventory, and shipment log."""
    return {
        "orders": copy.deepcopy(ORDERS),
        "inventory": copy.deepcopy(INVENTORY),
        "shipments": copy.deepcopy(SHIPMENTS),
    }


def load_state(path: str | Path) -> None:
    """Replace the catalog with the state stored in a JSON file."""
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    ORDERS.clear()
    ORDERS.update(state["orders"])
    INVENTORY.clear()
    INVENTORY.update(state["inventory"])
    SHIPMENTS.clear()
    SHIPMENTS.extend(state.get("shipments", []))


def _persist() -> None:
    """Write the current state back to STATE_FILE, when one is configured."""
    if STATE_FILE:
        Path(STATE_FILE).write_text(json.dumps(snapshot(), indent=2), encoding="utf-8")


if STATE_FILE:
    load_state(STATE_FILE)


def list_orders(status: Optional[str] = None) -> list[dict]:
    """List orders, optionally filtered by status."""
    orders = list(ORDERS.values())
    if status:
        orders = [order for order in orders if order["status"] == status]
    return orders


def get_order(order_id: str) -> dict:
    """Fetch a single order by ID."""
    order = ORDERS.get(order_id)
    if not order:
        return {"error": f"Order not found: {order_id}"}
    return order


def check_inventory(sku: str) -> dict:
    """Check on-hand inventory for a SKU."""
    item = INVENTORY.get(sku)
    if not item:
        return {"error": f"SKU not found: {sku}"}
    return item


def create_shipment(order_id: str, carrier: str = "UPS") -> dict:
    """Create a shipment for a paid order if inventory is available."""
    order = ORDERS.get(order_id)
    if not order:
        return {"error": f"Order not found: {order_id}"}
    if order["status"] != "paid":
        return {"error": f"Order {order_id} is {order['status']}; only paid orders can ship"}
    item = INVENTORY.get(order["sku"])
    if not item or item["on_hand"] < order["qty"]:
        return {"error": f"Insufficient inventory for {order['sku']}"}
    item["on_hand"] -= order["qty"]
    order["status"] = "shipped"
    shipment = {
        "shipment_id": f"SHP-{len(SHIPMENTS) + 1:04d}",
        "order_id": order_id,
        "carrier": carrier,
        "sku": order["sku"],
        "qty": order["qty"],
        "warehouse": item["warehouse"],
    }
    SHIPMENTS.append(shipment)
    _persist()
    return shipment


mcp.tool()(list_orders)
mcp.tool()(get_order)
mcp.tool()(check_inventory)
mcp.tool()(create_shipment)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8008"))
    host = os.environ.get("HOST", "0.0.0.0")
    mcp.run(transport="http", host=host, port=port)
