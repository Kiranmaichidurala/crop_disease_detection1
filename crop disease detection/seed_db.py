from app import db, Disease

def seed():
    db.create_all()
    entries = [
        {"name": "Banana_Panama_Disease", "description": "Fungal disease on banana", "prevention": "Use resistant varieties; avoid contaminated soil."},
        {"name": "Tomato_Late_Blight", "description": "Late blight affects tomatoes.", "prevention": "Remove infected plants; fungicide; crop rotation."},
        # add more manual entries as needed
    ]
    for e in entries:
        exists = Disease.query.filter_by(name=e['name']).first()
        if not exists:
            db.session.add(Disease(name=e['name'], description=e['description'], prevention=e['prevention']))
    db.session.commit()
    print("Seeded custom diseases")

if __name__ == "__main__":
    seed()