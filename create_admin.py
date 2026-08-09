from app import app, db, User
from werkzeug.security import generate_password_hash


with app.app_context():

    admin_email = "admin@farmerai.com"
    admin_password = "Admin@123"

    existing_admin = User.query.filter_by(
        email=admin_email
    ).first()

    if existing_admin:
        existing_admin.role = "admin"
        existing_admin.password = generate_password_hash(
            admin_password
        )

        db.session.commit()

        print("Admin account updated successfully!")

    else:
        admin = User(
            name="Farmer AI Admin",
            email=admin_email,
            password=generate_password_hash(
                admin_password
            ),
            role="admin"
        )

        db.session.add(admin)
        db.session.commit()

        print("Admin account created successfully!")

    print()
    print("Admin Email:", admin_email)
    print("Admin Password:", admin_password)