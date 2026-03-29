
import asyncio
import httpx


async def main() -> None:
    payload = {
        "pattern_name": "crown_shelf_right_spike",
        "min_age_days": 14,
        "max_age_days": 450,
        "top_k": 5,
        "max_coins_to_evaluate": 50,
        "vs_currency": "usd",
        "include_notes": True,
    }

    async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=30) as client:
        response = await client.post("/scan", json=payload)
        response.raise_for_status()
        print(response.json())


if __name__ == "__main__":
    asyncio.run(main())
