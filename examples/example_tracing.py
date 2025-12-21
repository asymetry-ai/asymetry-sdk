"""
Demonstration of trace hierarchy with LLM requests.
Shows how OpenAI calls are linked to parent spans.
"""

import time
import openai
from asymetry import (
    init_observability,
    observe,
    add_span_attribute,
    add_span_event,
    shutdown_observability,
)

# Initialize Asymetry
init_observability(service_name="trace-hierarchy-demo", enable_tracing=True, log_level="INFO")

# Setup OpenAI client
client = openai.OpenAI()


@observe()
def generate_product_description(product_name: str, category: str) -> dict:
    """
    Generate product description using OpenAI.

    Trace Hierarchy:
    - generate_product_description (parent span)
      └── openai.chat.completions.create (child LLM request)
    """
    add_span_attribute("product_name", product_name)
    add_span_attribute("category", category)

    add_span_event("llm_call_started")

    # This OpenAI call will be linked to the parent span
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You are a creative copywriter. Generate a compelling product description for {category} products.",
            },
            {"role": "user", "content": f"Write a product description for: {product_name}"},
        ],
        temperature=0.7,
        max_tokens=150,
    )

    add_span_event("llm_call_completed")

    description = response.choices[0].message.content

    return {
        "product_name": product_name,
        "description": description,
        "tokens_used": response.usage.total_tokens if response.usage else 0,
    }


@observe(
    name="process_customer_query", capture_args=True, attributes={"feature": "customer_support"}
)
def process_customer_query(user_id: str, query: str) -> dict:
    """
    Process customer query with multiple steps.

    Trace Hierarchy:
    - process_customer_query (root span)
      ├── validate_query (child span)
      ├── fetch_user_context (child span)
      └── generate_response (child span)
          └── openai.chat.completions.create (LLM request)
    """
    add_span_attribute("user_id", user_id)
    add_span_attribute("query_length", len(query))

    # Step 1: Validate query
    with_validation = validate_query(query)

    # Step 2: Fetch user context
    user_context = fetch_user_context(user_id)

    # Step 3: Generate AI response
    ai_response = generate_response(query, user_context)

    return {"user_id": user_id, "response": ai_response, "timestamp": time.time()}


@observe(name="validate_query")
def validate_query(query: str) -> bool:
    """Validate customer query."""
    add_span_event("validation_started")
    time.sleep(0.05)  # Simulate validation

    is_valid = len(query) > 10
    add_span_attribute("is_valid", is_valid)

    return is_valid


@observe(name="fetch_user_context", kind="client")
def fetch_user_context(user_id: str) -> dict:
    """Fetch user context from database."""
    add_span_event("database_query_started")
    time.sleep(0.1)  # Simulate DB query

    return {
        "user_id": user_id,
        "tier": "premium",
        "history": ["previous_query_1", "previous_query_2"],
    }


@observe(name="generate_response")
def generate_response(query: str, user_context: dict) -> str:
    """Generate AI response using OpenAI."""
    add_span_attribute("user_tier", user_context.get("tier"))
    add_span_event("ai_generation_started")

    # This OpenAI call will be nested under generate_response
    response = client.chat.completions.create(
        model="gpt-4-unknown",
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful customer support assistant. User tier: {user_context.get('tier')}",
            },
            {"role": "user", "content": query},
        ],
        temperature=0.5,
    )

    add_span_event("ai_generation_completed")

    return response.choices[0].message.content


@observe(name="batch_product_descriptions", capture_args=True)
def batch_product_descriptions(products: list[dict]) -> list[dict]:
    """
    Generate descriptions for multiple products.

    Trace Hierarchy:
    - batch_product_descriptions (root span)
      ├── generate_product_description (child span)
      │   └── openai.chat.completions.create (LLM request)
      ├── generate_product_description (child span)
      │   └── openai.chat.completions.create (LLM request)
      └── generate_product_description (child span)
          └── openai.chat.completions.create (LLM request)
    """
    add_span_attribute("product_count", len(products))
    add_span_event("batch_processing_started")

    results = []
    for i, product in enumerate(products):
        add_span_event(f"processing_product_{i+1}")

        result = generate_product_description(
            product_name=product["name"], category=product["category"]
        )
        results.append(result)

    add_span_event("batch_processing_completed")
    return results


def main():
    print("\n" + "=" * 80)
    print("Trace Hierarchy with LLM Requests - Live Demo")
    print("=" * 80 + "\n")

    print("IMPORTANT: This demo requires OpenAI API key to be set!")
    print("Set it with: export OPENAI_API_KEY='your-key-here'\n")

    try:
        # Example 1: Simple trace with single LLM call
        print("1. Simple Trace → LLM Call:")
        print("   Hierarchy: generate_product_description → openai.chat.completions.create")
        result1 = generate_product_description(
            product_name="Wireless Headphones", category="Electronics"
        )
        print(f"   ✅ Generated description (tokens: {result1.get('tokens_used', 0)})\n")

        # Example 2: Complex nested trace
        print("2. Complex Nested Trace:")
        print("   Hierarchy:")
        print("   process_customer_query (root)")
        print("   ├── validate_query")
        print("   ├── fetch_user_context")
        print("   └── generate_response")
        print("       └── openai.chat.completions.create (LLM)")
        result2 = process_customer_query(user_id="user_123", query="How do I reset my password?")
        print(f"   ✅ Query processed successfully\n")

        # Example 3: Batch processing with multiple LLM calls
        print("3. Batch Processing (Multiple LLM Calls):")
        print("   Hierarchy:")
        print("   batch_product_descriptions (root)")
        print("   ├── generate_product_description → openai.chat.completions.create")
        print("   ├── generate_product_description → openai.chat.completions.create")
        print("   └── generate_product_description → openai.chat.completions.create")

        products = [
            {"name": "Smart Watch", "category": "Wearables"},
            {"name": "Coffee Maker", "category": "Kitchen"},
            {"name": "Yoga Mat", "category": "Fitness"},
        ]
        result3 = batch_product_descriptions(products)
        print(f"   ✅ Generated {len(result3)} descriptions\n")

        print("=" * 80)
        print("✅ All examples completed!")
        print("\nIn your Asymetry dashboard, you'll see:")
        print("• Complete trace hierarchies")
        print("• LLM requests linked to parent spans")
        print("• trace_id and parent_span_id in LLM requests")
        print("• Query: 'Show all LLM calls in trace X'")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your OpenAI API key is set!")

    # Wait for export
    time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown_observability(timeout=5.0)
        print("✅ Shutdown complete\n")
