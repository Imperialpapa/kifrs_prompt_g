
import sys
import os
sys.path.append(os.getcwd()) # Assuming running from backend/

try:
    from database.supabase_client import supabase
except ImportError:
    # Try adding parent directory if running from backend subdir
    sys.path.append(os.path.dirname(os.getcwd()))
    from database.supabase_client import supabase

def check_table():
    try:
        print("Checking for rule_patterns table...")
        # Try to select from rule_patterns using admin client if possible
        response = supabase.table('rule_patterns').select('count', count='exact').limit(1).execute()
        print("Table 'rule_patterns' exists.")
        return True
    except Exception as e:
        print(f"Table check failed: {e}")
        return False

if __name__ == "__main__":
    if check_table():
        print("Migration seems to have been applied.")
    else:
        print("Migration needed.")
