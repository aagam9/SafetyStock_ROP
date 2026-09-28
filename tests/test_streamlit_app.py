import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_bundled_workflow_populates_analysis_session(self):
        app_path = Path(__file__).parents[1] / "streamlit_app.py"
        with patch("safety_stock_agent.API_KEY", None):
            app = AppTest.from_file(str(app_path))
            app.run(timeout=20)
            self.assertFalse(app.exception)
            app.radio[0].set_value("Use bundled sample data").run(timeout=20)
            run_button = next(button for button in app.button if button.label == "Run analysis")
            run_button.click().run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["analysis_results"]), 60)
        self.assertEqual(app.session_state["item_master"]["supplier"].nunique(), 6)
        self.assertEqual(len(app.session_state["assistant_data_model"].sku_view), 60)
        self.assertEqual(app.session_state["inventory_chat"], [])
        self.assertTrue(any(header.value == "Ask Your Inventory Data" for header in app.subheader))


if __name__ == "__main__":
    unittest.main()
