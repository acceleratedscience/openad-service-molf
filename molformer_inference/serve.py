from molformer_inference.molformer_implementation import MolformerRegression, MolformerMultitaskClassification, MolformerClassification

# register services
MolformerRegression.register()
MolformerClassification.register()
MolformerMultitaskClassification.register()

if __name__ == "__main__":
    from openad_service_utils import start_server
    # start the server
    start_server(port=8080)
