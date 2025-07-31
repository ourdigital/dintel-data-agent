#!/usr/bin/env python3
"""Test script for enhanced GA4 integration."""
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.connectors.google_analytics_enhanced import GoogleAnalyticsEnhancedConnector

def setup_logging():
    """Set up logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_enhanced_connector():
    """Test the enhanced Google Analytics connector."""
    print("🧪 Testing Enhanced Google Analytics Connector")
    print("=" * 50)
    
    try:
        # Initialize enhanced connector
        print("1. Initializing enhanced connector...")
        connector = GoogleAnalyticsEnhancedConnector()
        print("✅ Enhanced connector initialized successfully")
        
        # Test basic data fetching
        print("\n2. Testing basic data fetching...")
        df = connector.fetch_data(date_range="last_7_days")
        
        if df.empty:
            print("⚠️  No data returned (this might be expected)")
        else:
            print(f"✅ Data fetched successfully: {len(df)} rows")
            print(f"   Columns: {list(df.columns)}")
        
        # Test monthly report with Sheets integration
        print("\n3. Testing monthly report with Sheets integration...")
        try:
            monthly_result = connector.run_monthly_report(export_to_sheets=True)
            print(f"✅ Monthly report completed: {monthly_result['status']}")
            if monthly_result.get('sheets_result'):
                print(f"   Sheets export: {monthly_result['sheets_result']['status']}")
        except Exception as e:
            print(f"⚠️  Monthly report failed (environment may not be configured): {e}")
        
        # Test daily report with Sheets integration  
        print("\n4. Testing daily report with Sheets integration...")
        try:
            daily_result = connector.run_daily_report(days=3, export_to_sheets=True)
            print(f"✅ Daily report completed: {daily_result['status']}")
            if daily_result.get('sheets_result'):
                print(f"   Sheets export: {daily_result['sheets_result']['status']}")
        except Exception as e:
            print(f"⚠️  Daily report failed (environment may not be configured): {e}")
        
        print("\n🎉 Enhanced integration testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_environment_setup():
    """Test if environment is properly set up."""
    print("🔧 Testing Environment Setup")
    print("=" * 30)
    
    required_vars = [
        "SERVICE_ACCOUNT_FILE",
        "PROPERTY_ID", 
        "SHEET_ID"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
        else:
            print(f"✅ {var}: {os.getenv(var)[:20]}...")
    
    if missing_vars:
        print(f"\n⚠️  Missing environment variables: {missing_vars}")
        print("   Please create .env file or set these variables")
        return False
    
    print("\n✅ Environment setup looks good!")
    return True

def main():
    """Main test function."""
    setup_logging()
    
    print("🚀 Enhanced GA4 Integration Test Suite")
    print("=" * 40)
    
    # Test environment setup
    env_ok = test_environment_setup()
    
    print("\n")
    
    # Test enhanced connector
    if env_ok:
        test_enhanced_connector()
    else:
        print("⏭️  Skipping integration tests due to environment issues")
    
    print("\n" + "=" * 40)
    print("Test suite completed!")

if __name__ == "__main__":
    main()