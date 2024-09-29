class ResourceManager:
    def __init__(self):
        self.resources = []

    def acquire_resource(self, resource):
        self.resources.append(resource)

    def release_resources(self):
        for resource in self.resources:
            try:
                resource.close()
            except Exception as e:
                print(f"Error releasing resource: {e}")
        self.resources.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_resources()

# Usage
with ResourceManager() as manager:
    resource1 = open('file1.txt', 'r')
    manager.acquire_resource(resource1)
    resource2 = open('file2.txt', 'r')
    manager.acquire_resource(resource2)

# Resources are automatically released when exiting the context