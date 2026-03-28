import asyncio
import json
import random
from fastapi import WebSocket

POINTS = ["0", "15", "30", "40", "AD"]


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, match_id: str):
        await websocket.accept()
        if match_id not in self.active_connections:
            self.active_connections[match_id] = []
        self.active_connections[match_id].append(websocket)

    def disconnect(self, websocket: WebSocket, match_id: str):
        if match_id in self.active_connections:
            conns = self.active_connections[match_id]
            if websocket in conns:
                conns.remove(websocket)
            if not conns:
                del self.active_connections[match_id]

    async def broadcast_to_match(self, match_id: str, message: dict):
        conns = self.active_connections.get(match_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, match_id)

    async def broadcast_to_all(self, message: dict):
        for match_id in list(self.active_connections.keys()):
            msg = {**message, "match_id": match_id}
            await self.broadcast_to_match(match_id, msg)


manager = ConnectionManager()

# Shared game state across all connected matches
_game_state = {"p1_idx": 0, "p2_idx": 0, "server": 1}


async def simulate_match_pulse():
    """Continuously simulate live point-by-point updates and broadcast to all connected clients."""
    global _game_state
    while True:
        await asyncio.sleep(1.5)

        if not manager.active_connections:
            continue

        # Simulate a point being scored
        point_winner = random.choice([1, 2])
        if point_winner == 1:
            _game_state["p1_idx"] = min(_game_state["p1_idx"] + 1, len(POINTS) - 1)
        else:
            _game_state["p2_idx"] = min(_game_state["p2_idx"] + 1, len(POINTS) - 1)

        # Game over — reset points, flip server
        if _game_state["p1_idx"] == len(POINTS) - 1 or _game_state["p2_idx"] == len(POINTS) - 1:
            _game_state["p1_idx"] = 0
            _game_state["p2_idx"] = 0
            _game_state["server"] = 2 if _game_state["server"] == 1 else 1

        # Momentum: +1.0 = p1 dominating, -1.0 = p2 dominating
        momentum = round(random.uniform(-1.0, 1.0), 3)

        payload = {
            "type": "pulse",
            "server": _game_state["server"],
            "p1_points": POINTS[_game_state["p1_idx"]],
            "p2_points": POINTS[_game_state["p2_idx"]],
            "momentum": momentum,
        }

        await manager.broadcast_to_all(payload)


async def start_live_simulation(app):
    asyncio.create_task(simulate_match_pulse())
