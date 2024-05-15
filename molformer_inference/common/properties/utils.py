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
import ipaddress
import json
from typing import Any, Callable, Dict, List, Tuple, Type, Union

from rdkit import Chem

from modlamp.descriptors import GlobalDescriptor
from molformer_inference.common.domains.materials import MacroMolecule



# for proteins
def get_sequence(protein: MacroMolecule) -> str:
    """Safely returns an amino acid sequence of a macromolecule

    Args:
        protein: either an AA sequence or a rdkit.Chem.Mol object that can be converted to FASTA.

    Raises:
        TypeError: if the input was none of the above types.
        ValueError: if the sequence was empty or could not be parsed into FASTA.

    Returns:
        an AA sequence.
    """
    if isinstance(protein, str):
        seq = protein.upper().strip()
        return seq
    elif isinstance(protein, Chem.Mol):
        seq = Chem.MolToFASTA(protein).split()
    else:
        raise TypeError(f"Pass a string or rdkit.Chem.Mol object not {type(protein)}")
    if seq == []:
        raise ValueError(
            f"Sequence was empty or rdkit.Chem.Mol could not be converted: {protein}"
        )
    return seq[-1]


def get_descriptor(protein: MacroMolecule) -> GlobalDescriptor:
    """Convert a macromolecule to a modlamp GlobalDescriptor object.

    Args:
        protein: either an AA sequence or a rdkit.Chem.Mol object that can be converted to FASTA.

    Returns:
        GlobalDescriptor object.
    """
    seq = get_sequence(protein)
    return GlobalDescriptor(seq)


