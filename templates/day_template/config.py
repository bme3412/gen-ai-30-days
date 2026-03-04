"""
Industry Verticals Configuration

This file defines the 5 core industry verticals for the demo app.
Each vertical includes metadata, example use cases, and sample data.
To add a new vertical, simply add a new entry to the VERTICALS dict.
"""

# Day metadata - customize per day
DAY_CONFIG = {
    "day_number": 1,
    "title": "Agentic AI: Core Concepts & Design Patterns",
    "description": "Learn the foundational concepts and design patterns for building agentic AI systems.",
    "core_concept": "agentic_patterns",  # Used to load appropriate agent logic
}

# Industry verticals configuration
VERTICALS = {
    "financial": {
        "id": "financial",
        "name": "Financial Services",
        "icon": "chart-line",  # Font Awesome icon
        "emoji": "💰",
        "color": "#2E7D32",  # Green
        "gradient": "linear-gradient(135deg, #2E7D32 0%, #1B5E20 100%)",
        "description": "Banking, investments, insurance, and fintech applications",
        "examples": [
            "Portfolio risk analysis agent",
            "Fraud detection and alert system",
            "Loan approval workflow automation",
            "Regulatory compliance checker",
            "Market sentiment analyzer"
        ],
        "sample_prompts": [
            "Analyze the risk profile of this investment portfolio",
            "Flag suspicious transactions in this account activity",
            "Process this loan application and provide a recommendation"
        ],
        "sample_data": {
            "portfolio": {
                "stocks": ["AAPL", "GOOGL", "MSFT"],
                "allocation": [0.4, 0.35, 0.25],
                "total_value": 150000
            },
            "transaction": {
                "amount": 5000,
                "merchant": "Online Electronics Store",
                "location": "International",
                "time": "3:00 AM"
            }
        }
    },

    "healthcare": {
        "id": "healthcare",
        "name": "Healthcare",
        "icon": "hospital",
        "emoji": "🏥",
        "color": "#1976D2",  # Blue
        "gradient": "linear-gradient(135deg, #1976D2 0%, #0D47A1 100%)",
        "description": "Patient care, clinical operations, and health tech solutions",
        "examples": [
            "Patient triage assistant",
            "Clinical note extraction and summarization",
            "Insurance claim processing agent",
            "Appointment scheduling optimizer",
            "Drug interaction checker"
        ],
        "sample_prompts": [
            "Assess this patient's symptoms and recommend triage priority",
            "Extract key findings from this clinical note",
            "Process this insurance claim and identify any issues"
        ],
        "sample_data": {
            "patient": {
                "age": 45,
                "symptoms": ["chest pain", "shortness of breath", "fatigue"],
                "medical_history": ["hypertension", "diabetes"],
                "vitals": {"bp": "140/90", "hr": 88, "temp": 98.6}
            },
            "claim": {
                "procedure_code": "99213",
                "diagnosis": "E11.9",
                "amount": 250.00,
                "provider": "Primary Care Clinic"
            }
        }
    },

    "ecommerce": {
        "id": "ecommerce",
        "name": "E-commerce",
        "icon": "shopping-cart",
        "emoji": "🛒",
        "color": "#F57C00",  # Orange
        "gradient": "linear-gradient(135deg, #F57C00 0%, #E65100 100%)",
        "description": "Online retail, marketplaces, and digital commerce platforms",
        "examples": [
            "Product recommendation engine",
            "Inventory alert and reorder agent",
            "Customer support chatbot",
            "Returns processing workflow",
            "Dynamic pricing optimizer"
        ],
        "sample_prompts": [
            "Recommend products based on this customer's browsing history",
            "Analyze inventory levels and suggest reorder quantities",
            "Handle this customer complaint about a delayed shipment"
        ],
        "sample_data": {
            "customer": {
                "browsing_history": ["laptops", "monitors", "keyboards"],
                "purchase_history": ["mouse", "laptop stand"],
                "cart": ["webcam"],
                "segment": "tech_enthusiast"
            },
            "inventory": {
                "sku": "WC-HD-1080",
                "current_stock": 15,
                "reorder_point": 50,
                "lead_time_days": 7,
                "daily_sales_avg": 12
            }
        }
    },

    "real_estate": {
        "id": "real_estate",
        "name": "Real Estate",
        "icon": "home",
        "emoji": "🏠",
        "color": "#7B1FA2",  # Purple
        "gradient": "linear-gradient(135deg, #7B1FA2 0%, #4A148C 100%)",
        "description": "Property sales, rentals, and real estate technology",
        "examples": [
            "Property valuation estimator",
            "Lead qualification agent",
            "Lease document reviewer",
            "Market analysis reporter",
            "Showing schedule optimizer"
        ],
        "sample_prompts": [
            "Estimate the market value of this property",
            "Qualify this lead based on their requirements and budget",
            "Review this lease agreement and highlight key terms"
        ],
        "sample_data": {
            "property": {
                "address": "123 Main St, Austin, TX",
                "type": "Single Family",
                "sqft": 2400,
                "bedrooms": 4,
                "bathrooms": 2.5,
                "year_built": 2015,
                "lot_size": 0.25
            },
            "lead": {
                "name": "John Smith",
                "budget": 500000,
                "preferred_areas": ["Downtown", "East Side"],
                "timeline": "3 months",
                "pre_approved": True
            }
        }
    },

    "legal": {
        "id": "legal",
        "name": "Legal",
        "icon": "gavel",
        "emoji": "⚖️",
        "color": "#455A64",  # Blue Grey
        "gradient": "linear-gradient(135deg, #455A64 0%, #263238 100%)",
        "description": "Law firms, legal tech, and compliance solutions",
        "examples": [
            "Contract review and analysis agent",
            "Legal research assistant",
            "Due diligence document processor",
            "Compliance monitoring system",
            "Case document summarizer"
        ],
        "sample_prompts": [
            "Review this contract and identify potential risks",
            "Research precedents for this type of case",
            "Summarize the key findings from these due diligence documents"
        ],
        "sample_data": {
            "contract": {
                "type": "Service Agreement",
                "parties": ["Acme Corp", "Tech Solutions Inc"],
                "term": "24 months",
                "value": 150000,
                "key_clauses": ["indemnification", "termination", "IP rights"]
            },
            "case": {
                "type": "Contract Dispute",
                "jurisdiction": "Texas",
                "amount_in_controversy": 500000,
                "key_issues": ["breach of warranty", "consequential damages"]
            }
        }
    }
}

# Helper functions
def get_vertical(vertical_id: str) -> dict:
    """Get a vertical configuration by ID."""
    return VERTICALS.get(vertical_id)

def get_all_verticals() -> list:
    """Get all verticals as a list."""
    return list(VERTICALS.values())

def get_vertical_ids() -> list:
    """Get list of all vertical IDs."""
    return list(VERTICALS.keys())
