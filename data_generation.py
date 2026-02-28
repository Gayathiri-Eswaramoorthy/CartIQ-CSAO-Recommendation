import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from datetime import datetime, timedelta

# =====================
# CONFIG
# =====================

NUM_USERS = 10000
NUM_RESTAURANTS = 300
NUM_ITEMS = 2000
NUM_ORDERS = 500000

CITIES = ["Chennai", "Bangalore", "Mumbai", "Delhi", "Hyderabad"]
CUISINES = ["indian", "chinese", "italian"]
CATEGORIES = ["main", "side", "dessert", "beverage"]

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 10, 31)

np.random.seed(42)
random.seed(42)

# =====================
# USERS
# =====================

def generate_users():
    users = []
    for i in range(NUM_USERS):
        users.append({
            "user_id": i,
            "city": random.choice(CITIES),
            "budget_sensitivity": np.random.rand(),
            "veg_preference": np.random.rand(),
            "dessert_affinity": np.random.rand(),
            "beverage_affinity": np.random.rand(),
            "order_frequency": np.random.exponential(1.0)
        })
    return pd.DataFrame(users)

# =====================
# ITEMS
# =====================

def generate_items():
    items = []
    for i in range(NUM_ITEMS):
        items.append({
            "item_id": i,
            "cuisine": random.choice(CUISINES),
            "category": random.choice(CATEGORIES),
            "price": np.random.randint(100, 600),
            "veg_flag": np.random.choice([0, 1]),
        })
    return pd.DataFrame(items)

# =====================
# RESTAURANTS
# =====================

def generate_restaurants(items_df):
    restaurants = []
    menus = {}

    for r in range(NUM_RESTAURANTS):
        city = random.choice(CITIES)
        cuisine = random.choice(CUISINES)

        cuisine_items = items_df[items_df["cuisine"] == cuisine]["item_id"]
        menu_size = random.randint(20, 80)

        menu_items = cuisine_items.sample(menu_size, replace=False).tolist()

        restaurants.append({
            "restaurant_id": r,
            "city": city,
            "cuisine": cuisine,
            "price_tier": random.choice(["budget", "premium"]),
        })

        menus[r] = menu_items

    return pd.DataFrame(restaurants), menus

# =====================
# TIMESTAMP GENERATOR
# =====================

def random_timestamp():
    delta = END_DATE - START_DATE
    random_days = random.randint(0, delta.days)
    base_date = START_DATE + timedelta(days=random_days)

    # Bias toward lunch & dinner
    hour = random.choices(
        population=[12,13,14,19,20,21, random.randint(0,23)],
        weights=[3,3,3,3,3,3,1],
        k=1
    )[0]

    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return base_date.replace(hour=hour, minute=minute, second=second)

# =====================
# ORDER SIMULATION
# =====================

def simulate_orders(users_df, restaurants_df, items_df, menus):

    orders = []
    training_rows = []
    item_lookup = items_df.set_index("item_id").to_dict("index")

    for order_id in tqdm(range(NUM_ORDERS)):

        user = users_df.sample(1).iloc[0]

        # Select restaurant in same city
        city_restaurants = restaurants_df[
            restaurants_df["city"] == user["city"]
        ]
        restaurant = city_restaurants.sample(1).iloc[0]

        timestamp = random_timestamp()

        orders.append({
            "order_id": order_id,
            "user_id": user["user_id"],
            "restaurant_id": restaurant["restaurant_id"],
            "timestamp": timestamp
        })

        menu = menus[restaurant["restaurant_id"]]
        mains = [i for i in menu if item_lookup[i]["category"] == "main"]

        if not mains:
            continue

        cart = [random.choice(mains)]

        # 1–5 additional add-on attempts
        for step in range(random.randint(1, 5)):

            candidates = random.sample(menu, min(10, len(menu)))
            accepted_item = None

            for candidate in candidates:

                item = item_lookup[candidate]

                # Base probability
                base_prob = 0.08

                # Dessert affinity
                if item["category"] == "dessert":
                    base_prob += user["dessert_affinity"] * 0.4

                # Beverage affinity
                if item["category"] == "beverage":
                    base_prob += user["beverage_affinity"] * 0.4

                # Veg alignment
                if item["veg_flag"] == 1:
                    base_prob += user["veg_preference"] * 0.1

                # Price sensitivity
                if item["price"] > 400:
                    base_prob -= user["budget_sensitivity"] * 0.3

                # Penalize repeated categories
                existing_categories = [
                    item_lookup[i]["category"] for i in cart
                ]
                if item["category"] in existing_categories:
                    base_prob -= 0.15

                # Weekend boost
                if timestamp.weekday() >= 5:
                    base_prob += 0.05

                base_prob = max(0.01, min(base_prob, 0.9))

                if random.random() < base_prob:
                    accepted_item = candidate
                    break

            # Create ranking-ready rows
            for candidate in candidates:

                training_rows.append({
                    "order_id": order_id,
                    "user_id": user["user_id"],
                    "restaurant_id": restaurant["restaurant_id"],
                    "timestamp": timestamp,
                    "cart_state": cart.copy(),
                    "candidate_item": candidate,
                    "label": 1 if candidate == accepted_item else 0
                })

            if accepted_item is None:
                break

            cart.append(accepted_item)

    return pd.DataFrame(orders), pd.DataFrame(training_rows)

# =====================
# MAIN EXECUTION
# =====================

users_df = generate_users()
items_df = generate_items()
restaurants_df, menus = generate_restaurants(items_df)

orders_df, training_df = simulate_orders(
    users_df, restaurants_df, items_df, menus
)

users_df.to_csv("users.csv", index=False)
items_df.to_csv("items.csv", index=False)
restaurants_df.to_csv("restaurants.csv", index=False)
orders_df.to_csv("orders.csv", index=False)
training_df.to_csv("training_data.csv", index=False)

print("Dataset generation complete.")
