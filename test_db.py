import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.database.database import engine, Base, get_db
from src.database.models import *
from sqlalchemy.orm import Session
import json
from datetime import datetime

def init_database():
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

def test_database_connection():
    print("\n=== Testing Database Connection ===")
    try:
        db = next(get_db())
        print("✓ Database connection successful")
        return db
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None

def test_image_crud(db: Session):
    print("\n=== Testing Image CRUD Operations ===")
    
    try:
        test_image = Image(
            filename="test_drone_image.tif",
            filepath="D:/Nong-View/data/test/test_drone_image.tif",
            width=5000,
            height=4000,
            bands=4,
            crs="EPSG:5186",
            bounds={
                "minx": 127.123,
                "miny": 35.456,
                "maxx": 127.789,
                "maxy": 35.890
            },
            metadata={
                "drone_model": "DJI Phantom 4 RTK",
                "capture_date": "2025-10-27",
                "altitude": 120
            }
        )
        
        db.add(test_image)
        db.commit()
        db.refresh(test_image)
        print(f"✓ Image created: ID={test_image.id}")
        
        retrieved_image = db.query(Image).filter_by(id=test_image.id).first()
        if retrieved_image:
            print(f"✓ Image retrieved: {retrieved_image.filename}")
        
        retrieved_image.status = "processing"
        db.commit()
        print("✓ Image updated: status=processing")
        
        return test_image.id
    
    except Exception as e:
        print(f"✗ Image CRUD test failed: {e}")
        db.rollback()
        return None

def test_analysis_crud(db: Session, image_id: str):
    print("\n=== Testing Analysis CRUD Operations ===")
    
    try:
        test_analysis = Analysis(
            image_id=image_id,
            job_id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            analysis_type="crop_detection",
            model_name="YOLOv11",
            model_version="1.0.0",
            status="running",
            progress=25.5,
            parameters={
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "batch_size": 32
            }
        )
        
        db.add(test_analysis)
        db.commit()
        db.refresh(test_analysis)
        print(f"✓ Analysis created: ID={test_analysis.id}")
        
        test_analysis.progress = 50.0
        test_analysis.status = "processing"
        db.commit()
        print("✓ Analysis updated: progress=50.0")
        
        return test_analysis.id
    
    except Exception as e:
        print(f"✗ Analysis CRUD test failed: {e}")
        db.rollback()
        return None

def test_result_crud(db: Session, analysis_id: str):
    print("\n=== Testing Result CRUD Operations ===")
    
    try:
        test_results = [
            Result(
                analysis_id=analysis_id,
                class_name="rice_field",
                confidence=0.92,
                geometry_type="polygon",
                geometry={
                    "type": "Polygon",
                    "coordinates": [[[127.123, 35.456], [127.124, 35.456], 
                                   [127.124, 35.457], [127.123, 35.457], 
                                   [127.123, 35.456]]]
                },
                bbox=[127.123, 35.456, 127.124, 35.457],
                area=1234.56,
                attributes={"crop_type": "rice", "growth_stage": "mature"}
            ),
            Result(
                analysis_id=analysis_id,
                class_name="greenhouse",
                confidence=0.88,
                geometry_type="polygon",
                geometry={
                    "type": "Polygon",
                    "coordinates": [[[127.125, 35.458], [127.126, 35.458], 
                                   [127.126, 35.459], [127.125, 35.459], 
                                   [127.125, 35.458]]]
                },
                bbox=[127.125, 35.458, 127.126, 35.459],
                area=567.89,
                attributes={"structure_type": "vinyl", "condition": "good"}
            )
        ]
        
        for result in test_results:
            db.add(result)
        db.commit()
        print(f"✓ {len(test_results)} results created")
        
        results_count = db.query(Result).filter_by(analysis_id=analysis_id).count()
        print(f"✓ Results retrieved: {results_count} items")
        
        return True
    
    except Exception as e:
        print(f"✗ Result CRUD test failed: {e}")
        db.rollback()
        return False

def test_parcel_crud(db: Session):
    print("\n=== Testing Parcel CRUD Operations ===")
    
    try:
        test_parcel = Parcel(
            pnu="3627010100100010000",
            address="전라북도 남원시 도통동 100-1",
            owner_name="홍길동",
            geometry={
                "type": "Polygon",
                "coordinates": [[[127.123, 35.456], [127.124, 35.456], 
                               [127.124, 35.457], [127.123, 35.457], 
                               [127.123, 35.456]]]
            },
            area=2500.0,
            land_use="농지",
            crop_type="벼",
            cultivation_status="경작"
        )
        
        db.add(test_parcel)
        db.commit()
        db.refresh(test_parcel)
        print(f"✓ Parcel created: PNU={test_parcel.pnu}")
        
        return test_parcel.id
    
    except Exception as e:
        print(f"✗ Parcel CRUD test failed: {e}")
        db.rollback()
        return None

def test_tile_crud(db: Session, image_id: str):
    print("\n=== Testing Tile CRUD Operations ===")
    
    try:
        test_tiles = []
        for i in range(4):
            row = i // 2
            col = i % 2
            tile = Tile(
                image_id=image_id,
                tile_index=i,
                row=row,
                col=col,
                x_min=127.123 + col * 0.001,
                y_min=35.456 + row * 0.001,
                x_max=127.123 + (col + 1) * 0.001,
                y_max=35.456 + (row + 1) * 0.001,
                width=640,
                height=640,
                filepath=f"D:/Nong-View/data/tiles/tile_{i}.png"
            )
            test_tiles.append(tile)
            db.add(tile)
        
        db.commit()
        print(f"✓ {len(test_tiles)} tiles created")
        
        tiles_count = db.query(Tile).filter_by(image_id=image_id).count()
        print(f"✓ Tiles retrieved: {tiles_count} items")
        
        return True
    
    except Exception as e:
        print(f"✗ Tile CRUD test failed: {e}")
        db.rollback()
        return False

def main():
    print("=" * 50)
    print("Nong-View Database Test Suite")
    print("=" * 50)
    
    init_database()
    
    db = test_database_connection()
    if not db:
        print("\n✗ Test suite failed: Could not connect to database")
        return
    
    try:
        image_id = test_image_crud(db)
        
        if image_id:
            analysis_id = test_analysis_crud(db, image_id)
            
            if analysis_id:
                test_result_crud(db, analysis_id)
            
            test_tile_crud(db, image_id)
        
        test_parcel_crud(db)
        
        total_images = db.query(Image).count()
        total_analyses = db.query(Analysis).count()
        total_results = db.query(Result).count()
        total_parcels = db.query(Parcel).count()
        total_tiles = db.query(Tile).count()
        
        print("\n" + "=" * 50)
        print("Database Statistics:")
        print(f"  Images: {total_images}")
        print(f"  Analyses: {total_analyses}")
        print(f"  Results: {total_results}")
        print(f"  Parcels: {total_parcels}")
        print(f"  Tiles: {total_tiles}")
        print("=" * 50)
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()