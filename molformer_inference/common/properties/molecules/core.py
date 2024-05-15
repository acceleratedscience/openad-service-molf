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
import os
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml

# https://huggingface.co/ibm/MoLFormer-XL-both-10pct
from transformers import AutoModel, AutoTokenizer

try:
    from gt4sd_molformer.finetune.finetune_pubchem_light import (
        LightningModule as RegressionLightningModule,
    )
    from gt4sd_molformer.finetune.finetune_pubchem_light import (
        PropertyPredictionDataModule as RegressionDataModule,
    )
    from gt4sd_molformer.finetune.finetune_pubchem_light import (
        PropertyPredictionDataset as RegressionDataset,
    )
    from gt4sd_molformer.finetune.finetune_pubchem_light_classification import (
        LightningModule as ClassificationLightningModule,
    )
    from gt4sd_molformer.finetune.finetune_pubchem_light_classification import (
        PropertyPredictionDataModule as ClassificationDataModule,
    )
    from gt4sd_molformer.finetune.finetune_pubchem_light_classification import (
        PropertyPredictionDataset as ClassificationDataset,
    )
    from gt4sd_molformer.finetune.finetune_pubchem_light_classification_multitask import (
        MultitaskEmbeddingDataset,
        MultitaskModel,
        PropertyPredictionDataModule,
    )
    from gt4sd_molformer.finetune.ft_tokenizer.ft_tokenizer import MolTranBertTokenizer
except:  # noqa: E722
    pass


from pydantic.v1 import Field

# from tdc import Oracle
# from tdc.metadata import download_receptor_oracle_nam

from molformer_inference.common.algorithms.core import (
    ConfigurablePropertyAlgorithmConfiguration,
    Predictor,
    PredictorAlgorithm,
)

from molformer_inference.common.properties.core import (
    ApiTokenParameters,
    CallablePropertyPredictor,
    ConfigurableCallablePropertyPredictor,
    DomainSubmodule,
    IpAdressParameters,
    PropertyPredictorParameters,
    PropertyValue,
    S3Parameters,
)


# NOTE: property prediction parameters


class S3ParametersMolecules(S3Parameters):
    domain: DomainSubmodule = DomainSubmodule("molecules")


class MolformerParameters(S3ParametersMolecules):
    algorithm_name: str = "molformer"
    batch_size: int = Field(description="Prediction batch size", default=128)
    workers: int = Field(description="Number of data loading workers", default=8)
    device: Optional[str] = Field(description="Device to be used for inference", default=None)


class MolformerClassificationParameters(MolformerParameters):
    algorithm_application: str = "classification"


class MolformerMultitaskClassificationParameters(MolformerParameters):
    algorithm_application: str = "multitask_classification"


class MolformerRegressionParameters(MolformerParameters):
    algorithm_application: str = "regression"


class _Molformer(PredictorAlgorithm):
    """Base class for all Molformer predictive algorithms."""

    def __init__(self, parameters: MolformerParameters):
        # Set up the configuration from the parameters
        configuration = ConfigurablePropertyAlgorithmConfiguration(
            algorithm_type=parameters.algorithm_type,
            domain=parameters.domain,
            algorithm_name=parameters.algorithm_name,
            algorithm_application=parameters.algorithm_application,
            algorithm_version=parameters.algorithm_version,
        )

        self.batch_size = parameters.batch_size
        self.workers = parameters.workers

        self.transformer_model = "ibm/MoLFormer-XL-both-10pct"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The parent constructor calls `self.get_model`.
        super().__init__(configuration=configuration)

    def get_resources_path_and_config(self, resources_path: str):
        model_path = os.path.join(resources_path, "model.ckpt")
        config_path = os.path.join(resources_path, "hparams.yaml")

        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

        return config, model_path
    
    def get_model(self):
        return AutoModel.from_pretrained(self.transformer_model, deterministic_eval=True, trust_remote_code=True)

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.transformer_model, trust_remote_code=True)


class MolformerClassification(_Molformer):
    """Class for all Molformer classification algorithms."""

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """
        model = super().get_model()
        model.to(self.device)
        model.eval()

        # Wrapper to get the predictions
        def informative_model(samples: Union[str, List[str]]) -> List[float]:
            if isinstance(samples, str):
                samples = [samples]

            df = pd.DataFrame.from_dict({"smiles": samples})

            dataset = ClassificationDataset(df)
            datamodule = ClassificationDataModule(config, tokenizer)
            datamodule.test_ds = dataset

            preds = []
            for batch in datamodule.test_dataloader():
                with torch.no_grad():
                    batch = [x.to(self.device) for x in batch]
                    batch_output = model.testing_step(batch, 0, 0)

                preds_cpu = batch_output["pred"][:, 1]
                y_pred = np.where(preds_cpu >= 0.5, 1, 0)
                preds += y_pred.tolist()

            return preds

        return informative_model


class MolformerMultitaskClassification(_Molformer):
    """Class for all Molformer multitask classification algorithms."""

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """

        config, model_path = self.get_resources_path_and_config(resources_path)

        config["num_workers"] = 0

        tokenizer = self.get_tokenizer()

        model = MultitaskModel(config, tokenizer).load_from_checkpoint(
            model_path,
            strict=False,
            config=config,
            tokenizer=tokenizer,
            vocab=len(tokenizer.vocab),
        )

        model.to(self.device)
        model.eval()

        # Wrapper to get the predictions
        def informative_model(samples: Union[str, List[str]]) -> List[str]:
            if isinstance(samples, str):
                samples = [samples]

            df = pd.DataFrame.from_dict({"smiles": samples})

            dataset = MultitaskEmbeddingDataset(df)
            datamodule = PropertyPredictionDataModule(config, tokenizer)
            datamodule.test_ds = dataset

            preds = []
            for batch in datamodule.test_dataloader():
                with torch.no_grad():
                    batch = [x.to(self.device) for x in batch]
                    batch_output = model.testing_step(batch, 0, 0)

                batch_preds_idx = torch.argmax(batch_output["pred"], dim=1)
                batch_preds = [config["measure_names"][i] for i in batch_preds_idx]
                preds += batch_preds

            return preds

        return informative_model


class MolformerRegression(_Molformer):
    """Class for all Molformer regression algorithms."""

    def get_model(self, resources_path: str) -> Predictor:
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """

        config, model_path = self.get_resources_path_and_config(resources_path)

        config["num_workers"] = 0

        tokenizer = self.get_tokenizer()

        model = RegressionLightningModule(config, tokenizer).load_from_checkpoint(
            model_path,
            strict=False,
            config=config,
            tokenizer=tokenizer,
            vocab=len(tokenizer.vocab),
        )

        model.to(self.device)
        model.eval()

        # Wrapper to get the predictions
        def informative_model(samples: Union[str, List[str]]) -> List[float]:
            if isinstance(samples, str):
                samples = [samples]

            df = pd.DataFrame.from_dict({"smiles": samples})

            dataset = RegressionDataset(df, False, config["aug"])
            datamodule = RegressionDataModule(config, tokenizer)
            datamodule.test_ds = dataset

            preds = []
            for batch in datamodule.test_dataloader():
                with torch.no_grad():
                    batch = [x.to(self.device) for x in batch]
                    batch_output = model.testing_step(batch, 0, 0)

                preds += batch_output["pred"].view(-1).tolist()

            return preds

        return informative_model
