import argparse
import asyncio
from typing import Any
from memoraith import profile_model, set_output_path
from memoraith.config import config
from memoraith.exceptions import MemoraithError

async def main() -> None:
    parser = argparse.ArgumentParser(description="Memoraith: Lightweight Model Profiler")
    parser.add_argument("module", help="Python module containing the model and training function")
    parser.add_argument("function", help="Name of the function to profile")
    parser.add_argument("--output", default="memoraith_reports", help="Output directory for profiling results")
    parser.add_argument("--memory", action="store_true", help="Enable memory profiling")
    parser.add_argument("--computation", action="store_true", help="Enable computation time profiling")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU profiling")
    parser.add_argument("--real-time", action="store_true", help="Enable real-time visualization")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--report-format", choices=['html', 'pdf'], default='html', help="Report format")

    args = parser.parse_args()

    try:
        if args.config:
            config.load_from_file(args.config)

        set_output_path(args.output)

        module = __import__(args.module)
        func = getattr(module, args.function)

        @profile_model(memory=args.memory, computation=args.computation, gpu=args.gpu,
                       real_time_viz=args.real_time, report_format=args.report_format)
        async def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return await asyncio.to_thread(func, *args, **kwargs)

        await wrapped_func()

    except ImportError as e:
        print(f"Error: Could not import module '{args.module}'. {str(e)}")
    except AttributeError as e:
        print(f"Error: Function '{args.function}' not found in module '{args.module}'. {str(e)}")
    except MemoraithError as e:
        print(f"Memoraith Error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())