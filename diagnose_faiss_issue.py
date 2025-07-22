#!/usr/bin/env python3
"""
FAISS Search Diagnostic Script
Helps diagnose why FAISS is returning 0 results in IIS environment.
"""

import os
import sqlite3
import sys
from pathlib import Path

def check_database():
    """Check database contents"""
    print("=== DATABASE CHECK ===")
    
    db_path = "meeting_documents.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return False
    
    print(f"✅ Database found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        print(f"📄 Documents: {doc_count}")
        
        # Check chunks
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        print(f"📝 Chunks: {chunk_count}")
        
        if chunk_count > 0:
            # Show sample chunks
            cursor.execute("SELECT chunk_id, LENGTH(content), document_id FROM chunks LIMIT 3")
            samples = cursor.fetchall()
            print("📋 Sample chunks:")
            for chunk_id, content_len, doc_id in samples:
                print(f"  - {chunk_id[:20]}... (content: {content_len} chars, doc: {doc_id[:8]}...)")
        
        conn.close()
        return chunk_count > 0
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def check_faiss_index():
    """Check FAISS index"""
    print("\n=== FAISS INDEX CHECK ===")
    
    index_path = "vector_index.faiss"
    if not os.path.exists(index_path):
        print(f"❌ FAISS index not found: {index_path}")
        return False
    
    print(f"✅ FAISS index found: {index_path}")
    
    try:
        import faiss
        index = faiss.read_index(index_path)
        
        print(f"📊 Index type: {type(index).__name__}")
        print(f"📏 Vector count: {index.ntotal}")
        print(f"📐 Dimension: {index.d}")
        
        if isinstance(index, faiss.IndexIDMap):
            print("✅ Using IndexIDMap (supports direct deletion)")
        else:
            print("⚠️  Using old index format")
        
        return index.ntotal > 0
        
    except ImportError:
        print("❌ FAISS not available (missing faiss-cpu package)")
        return False
    except Exception as e:
        print(f"❌ FAISS error: {e}")
        return False

def test_hash_function():
    """Test the hash function used for chunk ID mapping"""
    print("\n=== HASH FUNCTION TEST ===")
    
    try:
        # Test with sample chunk IDs from database
        conn = sqlite3.connect("meeting_documents.db")
        cursor = conn.cursor()
        cursor.execute("SELECT chunk_id FROM chunks LIMIT 5")
        chunk_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if not chunk_ids:
            print("❌ No chunk IDs to test")
            return
        
        print("🧮 Testing hash function on sample chunk IDs:")
        
        for chunk_id in chunk_ids:
            # Same hash function used in vector_operations.py
            int_id = abs(hash(chunk_id)) % (2**31 - 1)
            print(f"  {chunk_id[:30]}... -> {int_id}")
        
        # Test for collisions
        int_ids = [abs(hash(cid)) % (2**31 - 1) for cid in chunk_ids]
        if len(int_ids) == len(set(int_ids)):
            print("✅ No hash collisions in sample")
        else:
            print("⚠️  Hash collisions detected in sample!")
            
    except Exception as e:
        print(f"❌ Hash test error: {e}")

def check_file_structure():
    """Check file structure"""
    print("\n=== FILE STRUCTURE CHECK ===")
    
    critical_files = [
        "meeting_documents.db",
        "vector_index.faiss", 
        "src/database/vector_operations.py",
        "flask_app.py"
    ]
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")

def suggest_fixes():
    """Suggest potential fixes"""
    print("\n=== SUGGESTED FIXES ===")
    
    print("1. 🔄 Restart IIS Application Pool")
    print("   - This will trigger fresh initialization of FAISS mappings")
    
    print("\n2. 🔍 Check IIS Logs")
    print("   - Look for 'Starting FAISS search' log messages")
    print("   - Check if 'ID mappings available' shows 0 mappings")
    
    print("\n3. 🛠️ Enable Debug Logging")
    print("   - The enhanced vector_operations.py now has detailed logging")
    print("   - Look for 'CRITICAL: int_to_chunk_id_map is empty' messages")
    
    print("\n4. 🗂️ Verify Database Permissions")
    print("   - Ensure IIS can read meeting_documents.db")
    print("   - Check file permissions in IIS environment")
    
    print("\n5. 🚨 Emergency Fix (if all else fails)")
    print("   - Delete vector_index.faiss to force rebuild")
    print("   - This will recreate proper ID mappings from database")

def main():
    print("FAISS Search Diagnostic Tool")
    print("=" * 50)
    
    # Run all checks
    db_ok = check_database()
    faiss_ok = check_faiss_index()
    
    if db_ok and faiss_ok:
        test_hash_function()
    
    check_file_structure()
    suggest_fixes()
    
    print("\n" + "=" * 50)
    if db_ok and faiss_ok:
        print("🟢 Core components look good - issue likely in ID mappings")
        print("💡 Try restarting the IIS application pool first")
    else:
        print("🔴 Core components have issues - see suggestions above")

if __name__ == "__main__":
    main()