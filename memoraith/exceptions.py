class MemoraithError(Exception):
    """Base exception class for Memoraith."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class FrameworkNotSupportedError(MemoraithError):
    """Exception raised when an unsupported framework is used."""
    def __init__(self, framework_name: str):
        self.framework_name = framework_name
        message = f"Framework '{framework_name}' is not supported."
        super().__init__(message)

class ConfigurationError(MemoraithError):
    """Exception raised when there's an issue with the configuration."""
    def __init__(self, config_item: str, details: str):
        self.config_item = config_item
        self.details = details
        message = f"Configuration error for {config_item}: {details}"
        super().__init__(message)

class ProfilingError(MemoraithError):
    """Exception raised when there's an error during profiling."""
    def __init__(self, component: str, details: str):
        self.component = component
        self.details = details
        message = f"Profiling error in {component}: {details}"
        super().__init__(message)



class DataCollectionError(MemoraithError):
    """Exception raised when there's an error collecting profiling data."""
    def __init__(self, data_type: str, details: str):
        self.data_type = data_type
        self.details = details
        message = f"Error collecting {data_type} data: {details}"
        super().__init__(message)

class AnalysisError(MemoraithError):
    """Exception raised when there's an error during data analysis."""
    def __init__(self, analysis_type: str, details: str):
        self.analysis_type = analysis_type
        self.details = details
        message = f"Error during {analysis_type} analysis: {details}"
        super().__init__(message)

class ReportGenerationError(MemoraithError):
    """Exception raised when there's an error generating reports."""
    def __init__(self, report_type: str, details: str):
        self.report_type = report_type
        self.details = details
        message = f"Error generating {report_type} report: {details}"
        super().__init__(message)

class GPUNotAvailableError(MemoraithError):
    """Exception raised when GPU profiling is requested but not available."""
    def __init__(self, details: str):
        self.details = details
        message = f"GPU profiling not available: {details}"
        super().__init__(message)
