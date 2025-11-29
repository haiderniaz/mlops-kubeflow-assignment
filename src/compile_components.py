"""
Script to compile Kubeflow pipeline components to YAML files.
"""

import os
from pathlib import Path
from kfp.dsl import component
from kfp import compiler

# Import components
from pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)


def compile_components():
    """Compile all pipeline components to YAML files."""
    
    # Create components directory if it doesn't exist
    components_dir = Path(__file__).parent.parent / "components"
    components_dir.mkdir(exist_ok=True)
    
    print(f"Compiling components to: {components_dir}")
    
    # List of components to compile
    components = [
        (data_extraction_component, "data_extraction"),
        (data_preprocessing_component, "data_preprocessing"),
        (model_training_component, "model_training"),
        (model_evaluation_component, "model_evaluation")
    ]
    
    # Compile each component
    for component_func, name in components:
        output_file = components_dir / f"{name}_component.yaml"
        
        try:
            compiler.Compiler().compile(
                pipeline_func=component_func,
                package_path=str(output_file)
            )
            print(f"✓ Compiled: {name}_component.yaml")
        except Exception as e:
            print(f"✗ Error compiling {name}: {str(e)}")
    
    print("\nComponent compilation completed!")
    print(f"YAML files saved in: {components_dir}")


if __name__ == "__main__":
    compile_components()
