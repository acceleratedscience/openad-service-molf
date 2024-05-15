#
# MIT License
#
# Copyright (c) 2022 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
from typing import Dict, Tuple, Type, Union
from molformer_inference.common.algorithms.core import PredictorAlgorithm
from molformer_inference.common.properties.core import PropertyPredictor, PropertyPredictorParameters
from molformer_inference.common.properties.molecules.core import (
    MolformerClassification,
    MolformerClassificationParameters,
    MolformerMultitaskClassification,
    MolformerMultitaskClassificationParameters,
    MolformerRegression,
    MolformerRegressionParameters,
)

# NOTE: all functions can be called with either a SMILES or a rdkit.Chem.Mol object.
MOLECULE_PROPERTY_PREDICTOR_FACTORY: Dict[
    str,
    Tuple[
        Union[Type[PropertyPredictor], Type[PredictorAlgorithm]],
        Type[PropertyPredictorParameters],
    ],
] = {
    "molformer_classification": (
        MolformerClassification,
        MolformerClassificationParameters,
    ),
    "molformer_multitask_classification": (
        MolformerMultitaskClassification,
        MolformerMultitaskClassificationParameters,
    ),
    "molformer_regression": (MolformerRegression, MolformerRegressionParameters),
}


AVAILABLE_MOLECULES_PROPERTY_PREDICTOR = sorted(MOLECULE_PROPERTY_PREDICTOR_FACTORY.keys())
