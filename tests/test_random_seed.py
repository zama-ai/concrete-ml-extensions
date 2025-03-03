import numpy as np
import pytest
import warnings

warnings.filterwarnings(
    "ignore",
    category=pytest.PytestRemovedIn9Warning,
    message="The \\(path: py.path.local\\) argument is deprecated",
)


def generate_random_data():
    """Generate some random data for testing."""
    return {
        "numpy_random": np.random.rand(5).tolist(),
        "numpy_integer": np.random.randint(1, 1000),
        "numpy_choice": np.random.choice(["a", "b", "c", "d", "e"], size=3).tolist(),
    }


def test_generate_random_data(tmp_path, request):
    """Test that generates random data and writes it to a file."""
    data = generate_random_data()
    seed = request.config.getoption("--randomly-seed")
    output_file = tmp_path / f"random_data_seed_{seed}.txt"
    if output_file.exists():
        output_file = tmp_path / f"random_data_seed_{seed}_bis.txt"
    with open(output_file, "w") as f:
        f.write(str(data))
    print(f"Data written to {output_file}")


def run_test_with_seed(seed, tmp_path):
    """Run the test_generate_random_data with a specific seed."""
    output_dir = tmp_path / f"test_run_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pytest.main(
        [
            "tests/test_random_seed.py::test_generate_random_data",
            f"--randomly-seed={seed}",
            f"--basetemp={output_dir}",
            "-s",
            "-q",
        ]
    )

    # Find the output file
    output_files = list(output_dir.rglob(f"random_data_seed_{seed}*.txt"))
    if not output_files:
        raise FileNotFoundError(f"No output file found for seed {seed}")
    return output_files[0]


def test_random_seed_consistency(tmp_path):
    """Test that the random seed produces consistent results across runs."""
    # Run the test twice with the same seed
    seed1 = 12345
    output_file1 = run_test_with_seed(seed1, tmp_path / "1")
    output_file2 = run_test_with_seed(seed1, tmp_path / "2")

    # Compare the outputs
    with open(output_file1, "r") as f1, open(output_file2, "r") as f2:
        print(f"Output file 1: {f1.read()}")
        print(f"Output file 2: {f2.read()}")
        assert f1.read() == f2.read(), "Outputs differ for the same seed"

    # Run the test with a different seed
    seed2 = 54321
    output_file3 = run_test_with_seed(seed2, tmp_path)

    # Compare the outputs (should be different)
    with open(output_file1, "r") as f1, open(output_file3, "r") as f3:
        print(f"Output file 3: {f3.read()}")
        assert f1.read() != f3.read(), "Outputs are the same for different seeds"

    print("\nAll random seed consistency checks passed successfully.")
