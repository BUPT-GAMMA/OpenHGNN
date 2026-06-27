def test_package_import_does_not_import_optional_model_dependencies():
    import openhgnn

    assert openhgnn.__version__ == "0.9.0"


def test_registry_validation_is_static_and_counts_v09_models():
    from openhgnn.registry import validate_registry

    report = validate_registry()

    assert report["ok"], report["issues"]
    assert report["counts"]["models"] >= 83
    assert report["counts"]["tasks"] >= 17
