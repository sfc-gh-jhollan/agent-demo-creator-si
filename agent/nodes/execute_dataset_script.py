import os
import subprocess


def execute_dataset_script(context, writer):
    writer("Executing dataset generation script...")

    output_directory = "generated_csvs"
    if os.path.exists(output_directory):
        for file in os.listdir(output_directory):
            file_path = os.path.join(output_directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(output_directory)
    os.makedirs(output_directory, exist_ok=True)

    script_path = "generated_script.py"
    if os.path.exists(script_path):
        os.remove(script_path)

    script_content = context["script"].strip("```python\n").strip("```")
    with open(script_path, "w") as script_file:
        script_file.write(script_content)

    context["output_directory"] = output_directory

    try:
        subprocess.run(["python", script_path], check=True)

        csv_files = [f for f in os.listdir(output_directory) if f.endswith(".csv")]
        context["csv_files"] = [os.path.join(output_directory, f) for f in csv_files]

    except subprocess.CalledProcessError as e:
        context["stack_trace"] = str(e)
        print(f"Error during script execution: {e}")
        return context

    return context
