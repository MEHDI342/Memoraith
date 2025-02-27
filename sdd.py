import os

def make_dir(directory_path):
    """
    Safely create a directory (and any missing parent directories).
    If it already exists, do nothing.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def make_file(file_path, contents=""):
    """
    Create an empty file (or with some default content) at the given path.
    If it already exists, do nothing.
    """
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(contents)

def main():
    # 1) Adjust the base directory path as needed:
    BASE_DIR = r"C:\Users\PC\Desktop\Megie\expat.me"

    # 2) Define the folder structure (relative to BASE_DIR).
    #    We'll store each directory as a list of (relative) paths.
    directories = [
        # Top-level
        "",
        "expat-me-frontend/public/locales",
        "expat-me-frontend/src/assets/images",
        "expat-me-frontend/src/assets/scss",
        "expat-me-frontend/src/components/common",
        "expat-me-frontend/src/components/auth",
        "expat-me-frontend/src/components/dashboard",
        "expat-me-frontend/src/components/admin",
        "expat-me-frontend/src/components/services",
        "expat-me-frontend/src/views",
        "expat-me-frontend/src/views/AdminViews",
        "expat-me-frontend/src/views/AuthViews",
        "expat-me-frontend/src/router",
        "expat-me-frontend/src/store",
        "expat-me-frontend/src/store/modules",
        "expat-me-frontend/src/store/plugins",
        "expat-me-frontend/src/services",
        "expat-me-frontend/src/utils",
        "expat-me-frontend/src/plugins",
        "expat-me-frontend/src/composables",
        "expat-me-frontend/src/constants",

        "expat-me-backend/src/config",
        "expat-me-backend/src/api/controllers",
        "expat-me-backend/src/api/middleware",
        "expat-me-backend/src/api/routes",
        "expat-me-backend/src/api/validations",
        "expat-me-backend/src/api/swagger",
        "expat-me-backend/src/models",
        "expat-me-backend/src/services",
        "expat-me-backend/src/utils",
        "expat-me-backend/src/db/repositories",
        "expat-me-backend/src/db/redis",
        "expat-me-backend/src/constants",
        "expat-me-backend/src/loaders",
        "expat-me-backend/src/jobs",
        "expat-me-backend/src/templates",
        "expat-me-backend/migrations",
        "expat-me-backend/seeds",
        "expat-me-backend/scripts",
        "expat-me-backend/tests",
        "expat-me-backend/tests/unit",
        "expat-me-backend/tests/integration",
        "expat-me-backend/tests/fixtures",
    ]

    # 3) Define the files that need to be created in each folder (if any).
    #    We use tuples (folder, file_name, optional_content).
    files = [
        # Top-level
        ("", ".gitignore", "# Git ignore file for Expat.me\n"),
        ("", "README.md", "# Expat.me Project\n"),
        ("", "LICENSE (optional)", ""),  # Or you can rename to LICENSE if you want

        # Frontend root
        ("expat-me-frontend", ".env", ""),
        ("expat-me-frontend", "package.json", "{\n  \"name\": \"expat-me-frontend\"\n}\n"),
        ("expat-me-frontend", "vite.config.js", "// Vite config placeholder\n"),

        # Frontend public
        ("expat-me-frontend/public", "favicon.ico", ""),
        ("expat-me-frontend/public", "index.html", "<!-- index.html placeholder -->\n"),
        ("expat-me-frontend/public/locales", "en.json", "{\n  \"greeting\": \"Hello\"\n}\n"),
        ("expat-me-frontend/public/locales", "fr.json", "{\n  \"greeting\": \"Bonjour\"\n}\n"),

        # Frontend scss
        ("expat-me-frontend/src/assets/scss", "_variables.scss", ""),
        ("expat-me-frontend/src/assets/scss", "_animations.scss", ""),
        ("expat-me-frontend/src/assets/scss", "_typography.scss", ""),
        ("expat-me-frontend/src/assets/scss", "main.scss", ""),

        # Common Components
        ("expat-me-frontend/src/components/common", "AppButton.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/common", "AppCard.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/common", "AppModal.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/common", "AppLoader.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/common", "AppAlert.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/common", "AppNavigation.vue", "<template></template>\n"),

        # Auth Components
        ("expat-me-frontend/src/components/auth", "LoginForm.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/auth", "RegisterForm.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/auth", "ForgotPasswordForm.vue", "<template></template>\n"),

        # Dashboard Components
        ("expat-me-frontend/src/components/dashboard", "RequestsList.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/dashboard", "RequestCard.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/dashboard", "RequestStatus.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/dashboard", "DocumentUploader.vue", "<template></template>\n"),

        # Admin Components
        ("expat-me-frontend/src/components/admin", "RequestsTable.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/admin", "UsersList.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/admin", "AdminDashboard.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/admin", "RequestDetails.vue", "<template></template>\n"),

        # Services Components
        ("expat-me-frontend/src/components/services", "ServiceCard.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/services", "ServiceForm.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/services", "PackageSelector.vue", "<template></template>\n"),
        ("expat-me-frontend/src/components/services", "PaymentForm.vue", "<template></template>\n"),

        # Views
        ("expat-me-frontend/src/views", "Home.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views", "About.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views", "Services.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views", "Contact.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views", "UserDashboard.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views", "RequestDetail.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views", "PaymentPage.vue", "<template></template>\n"),

        # AdminViews
        ("expat-me-frontend/src/views/AdminViews", "AdminDashboard.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views/AdminViews", "ManageRequests.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views/AdminViews", "ManageUsers.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views/AdminViews", "Settings.vue", "<template></template>\n"),

        # AuthViews
        ("expat-me-frontend/src/views/AuthViews", "Login.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views/AuthViews", "Register.vue", "<template></template>\n"),
        ("expat-me-frontend/src/views/AuthViews", "ForgotPassword.vue", "<template></template>\n"),

        # Router
        ("expat-me-frontend/src/router", "index.js", "// router index\n"),
        ("expat-me-frontend/src/router", "routes.js", "// route definitions\n"),
        ("expat-me-frontend/src/router", "guards.js", "// navigation guards\n"),
        ("expat-me-frontend/src/router", "auth-guard.js", "// auth guard\n"),

        # Store
        ("expat-me-frontend/src/store", "index.js", "// pinia or vuex store index\n"),
        ("expat-me-frontend/src/store/modules", "auth.js", ""),
        ("expat-me-frontend/src/store/modules", "users.js", ""),
        ("expat-me-frontend/src/store/modules", "requests.js", ""),
        ("expat-me-frontend/src/store/modules", "payments.js", ""),
        ("expat-me-frontend/src/store/modules", "notifications.js", ""),
        ("expat-me-frontend/src/store/plugins", "persist.js", "// store persistence plugin\n"),

        # Services
        ("expat-me-frontend/src/services", "api.js", ""),
        ("expat-me-frontend/src/services", "auth.service.js", ""),
        ("expat-me-frontend/src/services", "user.service.js", ""),
        ("expat-me-frontend/src/services", "request.service.js", ""),
        ("expat-me-frontend/src/services", "payment.service.js", ""),
        ("expat-me-frontend/src/services", "document.service.js", ""),

        # Utils
        ("expat-me-frontend/src/utils", "validators.js", ""),
        ("expat-me-frontend/src/utils", "formatters.js", ""),
        ("expat-me-frontend/src/utils", "animations.js", ""),
        ("expat-me-frontend/src/utils", "helpers.js", ""),

        # Plugins
        ("expat-me-frontend/src/plugins", "i18n.js", ""),
        ("expat-me-frontend/src/plugins", "gsap.js", ""),
        ("expat-me-frontend/src/plugins", "axios.js", ""),

        # Composables
        ("expat-me-frontend/src/composables", "useAuth.js", ""),
        ("expat-me-frontend/src/composables", "useNotifications.js", ""),
        ("expat-me-frontend/src/composables", "useRequests.js", ""),
        ("expat-me-frontend/src/composables", "useFormValidation.js", ""),

        # Constants
        ("expat-me-frontend/src/constants", "routes.js", ""),
        ("expat-me-frontend/src/constants", "apiEndpoints.js", ""),
        ("expat-me-frontend/src/constants", "servicePackages.js", ""),

        # App / main
        ("expat-me-frontend/src", "App.vue", "<template></template>\n"),
        ("expat-me-frontend/src", "main.js", "// main entry\n"),

        # Backend root
        ("expat-me-backend", ".env", ""),
        ("expat-me-backend", "package.json", "{\n  \"name\": \"expat-me-backend\"\n}\n"),
        ("expat-me-backend", "server.js", "// server entry\n"),
        ("expat-me-backend", "README.md", "# Expat.me Backend\n"),

        # Backend config
        ("expat-me-backend/src/config", "index.js", ""),
        ("expat-me-backend/src/config", "database.js", ""),
        ("expat-me-backend/src/config", "auth.js", ""),
        ("expat-me-backend/src/config", "email.js", ""),
        ("expat-me-backend/src/config", "stripe.js", ""),
        ("expat-me-backend/src/config", "redis.js", ""),
        ("expat-me-backend/src/config", "logger.js", ""),

        # Backend api/controllers
        ("expat-me-backend/src/api/controllers", "auth.controller.js", ""),
        ("expat-me-backend/src/api/controllers", "user.controller.js", ""),
        ("expat-me-backend/src/api/controllers", "request.controller.js", ""),
        ("expat-me-backend/src/api/controllers", "document.controller.js", ""),
        ("expat-me-backend/src/api/controllers", "payment.controller.js", ""),
        ("expat-me-backend/src/api/controllers", "admin.controller.js", ""),
        ("expat-me-backend/src/api/controllers", "notification.controller.js", ""),

        # Backend api/middleware
        ("expat-me-backend/src/api/middleware", "auth.middleware.js", ""),
        ("expat-me-backend/src/api/middleware", "validation.middleware.js", ""),
        ("expat-me-backend/src/api/middleware", "error.middleware.js", ""),
        ("expat-me-backend/src/api/middleware", "logger.middleware.js", ""),
        ("expat-me-backend/src/api/middleware", "cache.middleware.js", ""),
        ("expat-me-backend/src/api/middleware", "upload.middleware.js", ""),

        # Backend api/routes
        ("expat-me-backend/src/api/routes", "auth.routes.js", ""),
        ("expat-me-backend/src/api/routes", "user.routes.js", ""),
        ("expat-me-backend/src/api/routes", "request.routes.js", ""),
        ("expat-me-backend/src/api/routes", "document.routes.js", ""),
        ("expat-me-backend/src/api/routes", "payment.routes.js", ""),
        ("expat-me-backend/src/api/routes", "admin.routes.js", ""),
        ("expat-me-backend/src/api/routes", "notification.routes.js", ""),
        ("expat-me-backend/src/api/routes", "index.js", ""),

        # Backend api/validations
        ("expat-me-backend/src/api/validations", "auth.validation.js", ""),
        ("expat-me-backend/src/api/validations", "user.validation.js", ""),
        ("expat-me-backend/src/api/validations", "request.validation.js", ""),
        ("expat-me-backend/src/api/validations", "document.validation.js", ""),
        ("expat-me-backend/src/api/validations", "payment.validation.js", ""),

        # Backend api/swagger
        ("expat-me-backend/src/api/swagger", "auth.swagger.js", ""),
        ("expat-me-backend/src/api/swagger", "user.swagger.js", ""),
        ("expat-me-backend/src/api/swagger", "request.swagger.js", ""),
        ("expat-me-backend/src/api/swagger", "document.swagger.js", ""),
        ("expat-me-backend/src/api/swagger", "payment.swagger.js", ""),
        ("expat-me-backend/src/api/swagger", "index.js", ""),

        # Backend models
        ("expat-me-backend/src/models", "user.model.js", ""),
        ("expat-me-backend/src/models", "request.model.js", ""),
        ("expat-me-backend/src/models", "document.model.js", ""),
        ("expat-me-backend/src/models", "transaction.model.js", ""),
        ("expat-me-backend/src/models", "notification.model.js", ""),
        ("expat-me-backend/src/models", "task.model.js", ""),
        ("expat-me-backend/src/models", "country.model.js", ""),
        ("expat-me-backend/src/models", "service-package.model.js", ""),

        # Backend services
        ("expat-me-backend/src/services", "auth.service.js", ""),
        ("expat-me-backend/src/services", "user.service.js", ""),
        ("expat-me-backend/src/services", "request.service.js", ""),
        ("expat-me-backend/src/services", "document.service.js", ""),
        ("expat-me-backend/src/services", "payment.service.js", ""),
        ("expat-me-backend/src/services", "email.service.js", ""),
        ("expat-me-backend/src/services", "notification.service.js", ""),
        ("expat-me-backend/src/services", "cache.service.js", ""),
        ("expat-me-backend/src/services", "storage.service.js", ""),

        # Backend utils
        ("expat-me-backend/src/utils", "api-response.js", ""),
        ("expat-me-backend/src/utils", "jwt.js", ""),
        ("expat-me-backend/src/utils", "validators.js", ""),
        ("expat-me-backend/src/utils", "encryption.js", ""),
        ("expat-me-backend/src/utils", "formatter.js", ""),
        ("expat-me-backend/src/utils", "logger.js", ""),

        # Backend db
        ("expat-me-backend/src/db", "index.js", ""),
        ("expat-me-backend/src/db/repositories", "user.repository.js", ""),
        ("expat-me-backend/src/db/repositories", "request.repository.js", ""),
        ("expat-me-backend/src/db/repositories", "document.repository.js", ""),
        ("expat-me-backend/src/db/repositories", "transaction.repository.js", ""),
        ("expat-me-backend/src/db/repositories", "notification.repository.js", ""),
        ("expat-me-backend/src/db/redis", "index.js", ""),
        ("expat-me-backend/src/db/redis", "cache.js", ""),

        # Backend constants
        ("expat-me-backend/src/constants", "error-codes.js", ""),
        ("expat-me-backend/src/constants", "status-codes.js", ""),
        ("expat-me-backend/src/constants", "request-status.js", ""),
        ("expat-me-backend/src/constants", "roles.js", ""),

        # Backend loaders
        ("expat-me-backend/src/loaders", "express.js", ""),
        ("expat-me-backend/src/loaders", "database.js", ""),
        ("expat-me-backend/src/loaders", "redis.js", ""),
        ("expat-me-backend/src/loaders", "logger.js", ""),
        ("expat-me-backend/src/loaders", "index.js", ""),

        # Backend jobs
        ("expat-me-backend/src/jobs", "email-reminder.job.js", ""),
        ("expat-me-backend/src/jobs", "payment-check.job.js", ""),
        ("expat-me-backend/src/jobs", "scheduler.js", ""),

        # Backend templates
        ("expat-me-backend/src/templates", "welcome.template.js", ""),
        ("expat-me-backend/src/templates", "request-status.template.js", ""),
        ("expat-me-backend/src/templates", "document-request.template.js", ""),
        ("expat-me-backend/src/templates", "payment-confirmation.template.js", ""),

        # Backend app
        ("expat-me-backend/src", "app.js", ""),
    ]

    # 4) Create all directories
    for d in directories:
        full_path = os.path.join(BASE_DIR, d)
        make_dir(full_path)

    # 5) Create all files
    for folder, filename, content in files:
        folder_path = os.path.join(BASE_DIR, folder)
        file_path = os.path.join(folder_path, filename)
        make_file(file_path, content)

    print("Expat.me folder structure created successfully!")

if __name__ == '__main__':
    main()
