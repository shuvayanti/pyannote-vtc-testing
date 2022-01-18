import argparse
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from pyannote.audio import Inference
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.pipelines import MultilabelDetection
from pyannote.audio.tasks import VoiceTypeClassification
from pyannote.core import Annotation
from pyannote.database import FileFinder, get_protocol
from pyannote.database.util import load_rttm
from pyannote.pipeline import Optimizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


class BaseCommand:
    COMMAND = "command"
    DESCRIPTION = "Command description"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        pass

    @classmethod
    def run(cls, args: Namespace):
        pass


class TrainCommand(BaseCommand):
    COMMAND = "train"
    DESCRIPTION = "train the model"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("--classes", choices=["vtcdebug", "basalvoice"],
                            required=True,
                            type=str, help="Model architecture")
        parser.add_argument("--model_type", choices=["simple", "pyannet"],
                            required=True,
                            type=str, help="Model architecture")
        parser.add_argument("--restart", type=Path,
                            help="Continue with model path")
        parser.add_argument("--epoch", type=int, required=True,
                            help="Number of train epoch")

    @classmethod
    def run(cls, args: Namespace):
        protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
        if args.classes == "vtc_debug":
            classes_kwargs = {'classes': ["READER", "AGREER", "DISAGREER"],
                              'unions': {"COMMENTERS": ["AGREER", "DISAGREER"]}}
        else:
            classes_kwargs = {'classes': ["P", "NP"]}
        vtc = VoiceTypeClassification(protocol,
                                      **classes_kwargs,
                                      duration=2.00)
        if args.restart is None:
            if args.model_type == "simple":
                model = SimpleSegmentationModel(task=vtc)
            else:
                model = PyanNet(task=vtc)
        else:
            model = Model.from_pretrained(
                Path(args.restart),
                map_location=DEVICE,
                strict=False,
            )

        value_to_monitor, min_or_max = vtc.val_monitor

        checkpoints_path: Path = args.exp_dir / "checkpoints/"
        checkpoints_path.mkdir(parents=True, exist_ok=True)

        model_checkpoint = ModelCheckpoint(
            monitor=value_to_monitor,
            mode=min_or_max,
            save_top_k=5,
            every_n_epochs=1,
            save_last=True,
            dirpath=".",
            filename=f"{{epoch}}-{{{value_to_monitor}:.6f}}",
            verbose=True)

        early_stopping = EarlyStopping(
            monitor=value_to_monitor,
            mode=min_or_max,
            min_delta=0.0,
            patience=10.,
            strict=True,
            verbose=False)

        logger = TensorBoardLogger(args.exp_dir,
                                   name="VTCTest", version="", log_graph=False)

        trainer = Trainer(gpus=1, callbacks=[model_checkpoint, early_stopping], logger=logger)
        trainer.fit(model)


class TuneCommand(BaseCommand):
    COMMAND = "tune"
    DESCRIPTION = "tune the model hyperparameters using optuna"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Model checkpoint to tune pipeline with")
        parser.add_argument("-nit", "--n_iterations", type=int, default=50,
                            help="Number of tuning iterations")

    @classmethod
    def run(cls, args: Namespace):
        protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
        model = Inference(args.model_path)
        pipeline = MultilabelDetection(segmentation=model)
        validation_files = list(protocol.development())
        optimizer = Optimizer(pipeline)
        optimizer.tune(validation_files,
                       n_iterations=args.n_iterations,
                       show_progress=True)
        best_params = optimizer.best_params
        logging.info(f"Best params: \n{best_params}")
        params_filepath: Path = args.exp_dir / "best_params.yml"
        logging.info(f"Saving params to {params_filepath}")
        pipeline.instantiate(best_params)
        pipeline.dump_params(params_filepath)


class ApplyCommand(BaseCommand):
    COMMAND = "apply"
    DESCRIPTION = "apply the model on some data"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Model checkpoint to run pipeline with")
        parser.add_argument("--params", type=Path,
                            help="Path to best params. Default to EXP_DIR/best_params.yml")
        parser.add_argument("--apply_folder", type=Path,
                            help="Path to apply folder")

    @classmethod
    def run(cls, args: Namespace):
        protocol = get_protocol(args.protocol, preprocessors={"audio": FileFinder()})
        model = Inference(args.model_path)
        pipeline = MultilabelDetection(segmentation=model)
        pipeline.load_params(args.params)
        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        apply_folder.mkdir(parents=True, exist_ok=True)

        for file in tqdm(list(protocol.test())):
            logging.info(f"Inference for file {file['uri']}")
            annotation: Annotation = pipeline(file)
            with open(apply_folder / (file["uri"] + ".rttm"), "w") as rttm_file:
                annotation.write_rttm(rttm_file)


class ScoreCommand(BaseCommand):
    COMMAND = "score"
    DESCRIPTION = "score some inference"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("--apply_folder", type=Path,
                            help="Path to the inference files")
        parser.add_argument("--metric", choices=["fscore", "ier"],
                            default="fscore")

    @classmethod
    def run(cls, args: Namespace):
        protocol = get_protocol(args.protocol)
        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        annotations: Dict[str, Annotation] = {}
        for filepath in apply_folder.glob("*.rttm"):
            rttm_annots = load_rttm(filepath)
            annotations.update(rttm_annots)
        metric = None  # TODO: load either MultilabelIER or MultilabelFMeasure using task specs
        for file in protocol.test():
            if file["uri"] not in annotations:
                continue
            pass  # TODO : score


commands = [TrainCommand, TuneCommand, ApplyCommand, ScoreCommand]

argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="Show debug information in the standard output")
argparser.add_argument("exp_dir", type=Path,
                       help="Experimental folder")
subparsers = argparser.add_subparsers()

for command in commands:
    subparser = subparsers.add_parser(command.COMMAND)
    subparser.set_defaults(func=command.run,
                           command_class=command,
                           subparser=subparser)
    command.init_parser(subparser)

if __name__ == '__main__':
    args = argparser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if hasattr(args, "func"):
        args.func(args)
    else:
        argparser.print_help()
