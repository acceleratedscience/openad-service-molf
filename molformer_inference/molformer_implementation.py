import os
from typing import List, Union
import importlib_resources
import numpy as np
import pandas as pd
import torch
import yaml
from pydantic.v1 import Field

# molformer modules
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


# openad wrapper modules
from openad_service_utils import SimplePredictor, PredictorTypes, DomainSubmodule


class MolformerRegression(SimplePredictor):
    """Class for all Molformer regression algorithms."""
    # s3 params
    domain: DomainSubmodule = DomainSubmodule("molecules")
    algorithm_name: str = "molformer"
    algorithm_application: str = "regression"
    algorithm_version: str = "molformer_alpha_public_test"
    # other params
    property_type: str = PredictorTypes.MOLECULE
    # my model params
    batch_size: int = Field(description="Prediction batch size", default=128)
    workers: int = Field(description="Number of data loading workers", default=8)
    device: str = Field(description="Device to be used for inference", default="cpu")

    def get_resources_path_and_config(self, resources_path: str):
        model_path = os.path.join(resources_path, "model.ckpt")
        config_path = os.path.join(resources_path, "hparams.yaml")

        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

        return config, model_path

    def setup(self):
        """Setup Model"""
        self.model_path = self.get_model_location()
        print(f"\nRESOURCE PATH: {self.model_path}\n")
        self.config, model_path = self.get_resources_path_and_config(self.model_path)

        self.config["num_workers"] = 0

        self.tokenizer_path = importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"
        self.tokenizer = MolTranBertTokenizer(self.tokenizer_path)

        self.model = RegressionLightningModule.load_from_checkpoint(
            model_path,
            strict=False,
            config=self.config,
            tokenizer=self.tokenizer,
            vocab=len(self.tokenizer.vocab),
        )
        self.model.to(self.device)
        self.model.eval()

        # Wrapper to get the predictions
    def predict(self, samples: Union[str, List[str]]) -> List[float]:
        print(f">> {self.device=}")
        if isinstance(samples, str):
            samples = [samples]
        df = pd.DataFrame.from_dict({"smiles": samples})
        dataset = RegressionDataset(df, False, self.config["aug"])
        datamodule = RegressionDataModule(self.config, self.tokenizer)
        datamodule.test_ds = dataset
        preds = []
        for batch in datamodule.test_dataloader():
            with torch.no_grad():
                batch = [x.to(self.device) for x in batch]
                batch_output = self.model.testing_step(batch, 0, 0)
            preds += batch_output["pred"].view(-1).tolist()
        return preds


class MolformerClassification(SimplePredictor):
    """Class for all Molformer classification algorithms."""
    # s3 params
    domain: DomainSubmodule = DomainSubmodule("molecules")
    algorithm_name: str = "molformer"
    algorithm_application: str = "classification"
    algorithm_version: str = "molformer_bace_public_test"
    # other params
    property_type: str = PredictorTypes.MOLECULE
    # my model params
    batch_size: int = Field(description="Prediction batch size", default=128)
    workers: int = Field(description="Number of data loading workers", default=8)
    device: str = Field(description="Device to be used for inference", default="cpu")

    def get_resources_path_and_config(self, resources_path: str):
        model_path = os.path.join(resources_path, "model.ckpt")
        config_path = os.path.join(resources_path, "hparams.yaml")

        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)

        return config, model_path

    def setup(self):
        self.config, model_path = self.get_resources_path_and_config(self.get_model_location())

        self.config["num_workers"] = 0
        self.tokenizer_path = importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"
        self.tokenizer = MolTranBertTokenizer(self.tokenizer_path)

        self.model = ClassificationLightningModule.load_from_checkpoint(
            model_path,
            strict=False,
            config=self.config,
            tokenizer=self.tokenizer,
            vocab=len(self.tokenizer.vocab),
        )

        self.model.to(self.device)
        self.model.eval()

    # Wrapper to get the predictions
    def predict(self, samples: Union[str, List[str]]) -> List[float]:
        print(f">> {self.device=}")
        if isinstance(samples, str):
            samples = [samples]
        df = pd.DataFrame.from_dict({"smiles": samples})
        dataset = ClassificationDataset(df)
        datamodule = ClassificationDataModule(self.config, self.tokenizer)
        datamodule.test_ds = dataset
        preds = []
        for batch in datamodule.test_dataloader():
            with torch.no_grad():
                batch = [x.to(self.device) for x in batch]
                batch_output = self.model.testing_step(batch, 0, 0)
            preds_cpu = batch_output["pred"][:, 1]
            y_pred = np.where(preds_cpu >= 0.5, 1, 0)
            preds += y_pred.tolist()
        return preds


class MolformerMultitaskClassification(SimplePredictor):
    # s3 params
    """Class for all Molformer multitask classification algorithms."""
    domain: DomainSubmodule = DomainSubmodule("molecules")
    algorithm_name: str = "molformer"
    algorithm_application: str = "multitask_classification"
    algorithm_version: str = "molformer_clintox_test"
    # other params
    property_type: str = PredictorTypes.MOLECULE
    # my model params
    batch_size: int = Field(description="Prediction batch size", default=128)
    workers: int = Field(description="Number of data loading workers", default=8)
    device: str = Field(description="Device to be used for inference", default="cpu")
    
    def get_resources_path_and_config(self, resources_path: str):
        model_path = os.path.join(resources_path, "model.ckpt")
        config_path = os.path.join(resources_path, "hparams.yaml")

        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        self.tokenizer_path = importlib_resources.files("gt4sd_molformer") / "finetune/bert_vocab.txt"

        return config, model_path

    def setup(self):
        """Instantiate the actual model.

        Args:
            resources_path: local path to model files.

        Returns:
            Predictor: the model.
        """

        self.config, model_path = self.get_resources_path_and_config(self.get_model_location())

        self.config["num_workers"] = 0

        self.tokenizer = MolTranBertTokenizer(self.tokenizer_path)

        self.model = MultitaskModel.load_from_checkpoint(
            model_path,
            strict=False,
            config=self.config,
            tokenizer=self.tokenizer,
            vocab=len(self.tokenizer.vocab),
        )

        self.model.to(self.device)
        self.model.eval()

    # Wrapper to get the predictions
    def predict(self, samples: Union[str, List[str]]) -> List[str]:
        print(f">> {self.device=}")
        if isinstance(samples, str):
            samples = [samples]
        df = pd.DataFrame.from_dict({"smiles": samples})
        dataset = MultitaskEmbeddingDataset(df)
        datamodule = PropertyPredictionDataModule(self.config, self.tokenizer)
        datamodule.test_ds = dataset
        preds = []
        for batch in datamodule.test_dataloader():
            with torch.no_grad():
                batch = [x.to(self.device) for x in batch]
                batch_output = self.model.testing_step(batch, 0, 0)
            batch_preds_idx = torch.argmax(batch_output["pred"], dim=1)
            batch_preds = [self.config["measure_names"][i] for i in batch_preds_idx]
            preds += batch_preds
        return preds
