"""
Music Theory Domain for FluxEM.

Provides algebraic embeddings for:
- Pitch and intervals (frequency ratios)
- Chords and harmony (interval structures)
- Scales and modes (scale patterns)
- Atonal theory (12-tone sets as matrices)
- Rhythm and meter (time relationships)

All encoded using the mathematical structure of music theory.
"""

from .atonal import (
    AtonalSetEncoder,
    pitch_class_set_to_vector,
    normal_form,
    prime_form,
    interval_class_vector,
    forte_number,
    transposition,
    inversion,
    tin_operation,
    multiplication,
    interval_class_similarity,
    z_related,
    subset_of,
    row_matrix,
)

__all__ = [
    "PitchEncoder",
    "ChordEncoder",
    "ScaleEncoder",
    "AtonalSetEncoder",
    "pitch_class_set_to_vector",
    "normal_form",
    "prime_form",
    "interval_class_vector",
    "forte_number",
    "transposition",
    "inversion",
    "tin_operation",
    "multiplication",
    "interval_class_similarity",
    "z_related",
    "subset_of",
    "row_matrix",
]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from ...backend import get_backend

from ...core.base import (
    DOMAIN_TAGS,
    EMBEDDING_DIM,
    create_embedding,
    log_encode_value,
    log_decode_value,
)

# =============================================================================
# Music Theory Constants
# =============================================================================

# Note names
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# MIDI note numbers
MIDI_A4 = 69  # A4 = 69 in MIDI
FREQ_A4 = 440.0  # Standard tuning

# Intervals in semitones
INTERVALS = {
    "unison": 0,
    "minor second": 1,
    "major second": 2,
    "minor third": 3,
    "major third": 4,
    "fourth": 5,
    "tritone": 6,
    "fifth": 7,
    "minor sixth": 8,
    "major sixth": 9,
    "minor seventh": 10,
    "major seventh": 11,
    "octave": 12,
}

# Chord qualities (semitone pattern from root)
CHORD_PATTERNS = {
    "major": [0, 4, 7],  # Major triad
    "minor": [0, 3, 7],  # Minor triad
    "diminished": [0, 3, 6],  # Diminished triad
    "augmented": [0, 4, 8],  # Augmented triad
    "major 7": [0, 4, 7, 11],  # Major 7th
    "minor 7": [0, 3, 7, 10],  # Minor 7th
    "dominant 7": [0, 4, 7, 10],  # Dominant 7th
    "half-diminished 7": [0, 3, 6, 10],  # Half-diminished 7th
    "fully-diminished 7": [0, 3, 6, 9],  # Fully-diminished 7th
}

# Scale patterns (semitone intervals)
SCALE_PATTERNS = {
    "major": [0, 2, 4, 5, 7, 9, 11],  # Major scale
    "natural minor": [0, 2, 3, 5, 7, 8, 10],  # Natural minor
    "harmonic minor": [0, 2, 3, 5, 7, 8, 11],  # Harmonic minor
    "melodic minor": [0, 2, 3, 5, 7, 9, 11],  # Melodic minor (ascending)
    "pentatonic major": [0, 2, 4, 7, 9],  # Major pentatonic
    "pentatonic minor": [0, 3, 5, 7, 10],  # Minor pentatonic
    "blues": [0, 3, 5, 6, 7, 10],  # Blues scale
    "dorian": [0, 2, 3, 5, 7, 9, 10],  # Dorian mode
    "phrygian": [0, 1, 3, 5, 7, 8, 10],  # Phrygian mode
    "lydian": [0, 2, 4, 6, 7, 9, 11],  # Lydian mode
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],  # Mixolydian mode
    "locrian": [0, 1, 3, 5, 6, 8, 10],  # Locrian mode
}

# =============================================================================
# Pitch Encoding (Physics + Notation)
# =============================================================================

# Embedding layout for PITCH (dims 8-71):
# dims 0-1:   MIDI number (integer, normalized)
# dims 2-3:   Frequency (Hz, log encoded)
# dim 4:      Octave number
# dim 5:      Scale degree (0-6 in current key)
# dim 6:      Chromatic degree (0-11)
# dim 7:      Accidentals (-1: flat, 0: natural, 1: sharp)
# dims 8-15:  Harmonic series partials (first 8 harmonics)
# dims 16-23: Envelope (ADSR parameters)
# dims 24-31: Reserved for timbre

PITCH_MIDI_OFFSET = 0
PITCH_FREQ_OFFSET = 2
PITCH_OCTAVE_OFFSET = 4
PITCH_SCALE_DEGREE = 5
PITCH_CHROMATIC = 6
PITCH_ACCIDENTAL = 7
PITCH_HARMONIC_OFFSET = 8
PITCH_ENVELOPE_OFFSET = 16


def midi_to_freq(midi: int) -> float:
    """Convert MIDI note number to frequency (Hz)."""
    return FREQ_A4 * (2 ** ((midi - MIDI_A4) / 12.0))


def freq_to_midi(freq: float) -> float:
    """Convert frequency to MIDI note number."""
    return 69 + 12 * math.log2(freq / 440.0)


def midi_to_note_octave(midi: int) -> Tuple[str, int]:
    """Convert MIDI number to note name and octave."""
    octave = midi // 12 - 1
    note = NOTE_NAMES[midi % 12]
    return note, octave


class PitchEncoder:
    """
    Encoder for musical pitches with both physics and notation.

    Combines:
    - Frequency representation (physics)
    - Musical notation (note name, octave, accidental)
    - Harmonic content (sound)
    """

    domain_tag = DOMAIN_TAGS["music_pitch"]
    domain_name = "music_pitch"

    def encode(self, pitch: Union[int, str, Tuple[str, int]]) -> Any:
        """
        Encode a musical pitch.

        Args:
            pitch: Can be:
                - MIDI number (int, 0-127)
                - Note string ("A4", "C#5", "Bb3")
                - (note, octave) tuple (("A", 4))

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        # Parse input to MIDI number
        if isinstance(pitch, int):
            midi = pitch
        elif isinstance(pitch, str):
            midi = self._parse_note_string(pitch)
        elif isinstance(pitch, tuple):
            midi = self._parse_note_tuple(pitch)
        else:
            raise ValueError(f"Invalid pitch: {pitch}")

        midi = max(0, min(127, midi))  # Clamp to MIDI range

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # MIDI number (normalized)
        emb = backend.at_add(emb, 8 + PITCH_MIDI_OFFSET, midi / 127.0)

        # Frequency (log encoded)
        freq = midi_to_freq(midi)
        _, log_freq = log_encode_value(freq)
        emb = backend.at_add(emb, 8 + PITCH_FREQ_OFFSET, log_freq)

        # Octave
        octave = midi // 12 - 1
        emb = backend.at_add(emb, 8 + PITCH_OCTAVE_OFFSET, octave / 10.0)

        # Chromatic degree
        chromatic = midi % 12
        emb = backend.at_add(emb, 8 + PITCH_CHROMATIC, chromatic / 12.0)

        # Accidentals (derive from note name)
        note_name, _ = midi_to_note_octave(midi)
        if "#" in note_name:
            emb = backend.at_add(emb, 8 + PITCH_ACCIDENTAL, 1.0)  # Sharp
        elif "b" in note_name:
            emb = backend.at_add(emb, 8 + PITCH_ACCIDENTAL, -1.0)  # Flat
        else:
            emb = backend.at_add(emb, 8 + PITCH_ACCIDENTAL, 0.0)  # Natural

        # Harmonic series (first 8 partials)
        for i in range(8):
            harmonic_freq = freq * (i + 1)
            _, log_harmonic = log_encode_value(harmonic_freq)
            emb = backend.at_add(emb, 8 + PITCH_HARMONIC_OFFSET + i, log_harmonic / 50.0)

        return emb

    def decode(self, emb: Any) -> int:
        """
        Decode embedding to MIDI note number.

        Returns:
            MIDI note number (0-127)
        """
        midi_norm = emb[8 + PITCH_MIDI_OFFSET].item()
        midi = int(round(midi_norm * 127.0))
        return max(0, min(127, midi))

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid pitch."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Pitch operations (deterministic)
    # ========================================================================

    def transpose(self, emb: Any, semitones: int) -> Any:
        """Transpose a pitch by semitone interval."""
        midi_norm = emb[8 + PITCH_MIDI_OFFSET].item()
        midi = int(round(midi_norm * 127.0))

        # Transpose
        new_midi = midi + semitones
        new_midi = max(0, min(127, new_midi))

        # Create new embedding
        return self.encode(new_midi)

    def interval(self, emb1: Any, emb2: Any) -> int:
        """Calculate the interval between two pitches in semitones."""
        midi1 = int(round(emb1[8 + PITCH_MIDI_OFFSET].item() * 127.0))
        midi2 = int(round(emb2[8 + PITCH_MIDI_OFFSET].item() * 127.0))

        return midi2 - midi1

    def frequency_ratio(self, emb1: Any, emb2: Any) -> float:
        """Calculate the frequency ratio between two pitches."""
        log_freq1 = emb1[8 + PITCH_FREQ_OFFSET].item()
        log_freq2 = emb2[8 + PITCH_FREQ_OFFSET].item()

        # Ratio = freq2 / freq1 = exp(log_freq2 - log_freq1)
        return math.exp(log_freq2 - log_freq1)

    def in_same_key(self, emb1: Any, emb2: Any, key_root: int) -> bool:
        """
        Check if two pitches are in the same key.

        Args:
            key_root: MIDI number of key root (0-11)

        Returns:
            True if both pitches are diatonic to the key
        """
        midi1 = int(round(emb1[8 + PITCH_MIDI_OFFSET].item() * 127.0))
        midi2 = int(round(emb2[8 + PITCH_MIDI_OFFSET].item() * 127.0))

        # Get chromatic degrees relative to key
        degree1 = (midi1 - key_root) % 12
        degree2 = (midi2 - key_root) % 12

        # Major scale degrees: 0, 2, 4, 5, 7, 9, 11
        major_degrees = {0, 2, 4, 5, 7, 9, 11}

        return degree1 in major_degrees and degree2 in major_degrees

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _parse_note_string(self, note_str: str) -> int:
        """Parse note string like 'A4' or 'C#5' to MIDI."""
        note_str = note_str.strip()

        # Find octave number
        octave = 4
        for i, c in enumerate(note_str):
            if c.isdigit():
                octave = int(note_str[i:])
                note_part = note_str[:i]
                break
        else:
            note_part = note_str

        # Find note name
        note_name = note_part[0].upper()

        # Find accidental
        accidental = 0
        for c in note_part[1:]:
            if c == "#":
                accidental += 1
            elif c == "b":
                accidental -= 1

        # Calculate MIDI
        # Semitone offset from C (C=0, D=2, E=4, F=5, G=7, A=9, B=11)
        semitone_offset = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}.get(
            note_name, 0
        )

        # MIDI = (octave + 1) * 12 + semitone_offset + accidental
        # A4 = (4+1)*12 + 9 = 69 ✓
        midi = (octave + 1) * 12 + semitone_offset + accidental
        return midi

    def _parse_note_tuple(self, note_tuple: Tuple[str, int]) -> int:
        """Parse (note, octave) tuple to MIDI."""
        note, octave = note_tuple
        return self._parse_note_string(f"{note}{octave}")


# =============================================================================
# Chord Encoding (Harmony)
# =============================================================================

# Embedding layout for CHORD (dims 8-71):
# dims 0-1:   Root note MIDI number (normalized)
# dim 2:      Chord quality (encoded as integer 0-9)
# dim 3:      Inversion (0, 1, 2, or 3)
# dims 4-7:   Extension notes (7th, 9th, 11th, 13th)
# dims 8-15:  Voicing (which notes in the chord)
# dims 16-23: Interval pattern (semitones from root)
# dims 24-31: Tension/consonance scores
# dims 32-47: Harmonic series interaction
# dims 48-63: Function in key (tonic, dominant, etc.)

CHORD_ROOT_OFFSET = 0
CHORD_QUALITY_OFFSET = 2
CHORD_INVERSION = 3
CHORD_EXTENSION_OFFSET = 4
CHORD_VOICING_OFFSET = 8
CHORD_INTERVAL_OFFSET = 16
CHORD_TENSION_OFFSET = 24
CHORD_FUNCTION_OFFSET = 32


class ChordEncoder:
    """
    Encoder for musical chords.

    Captures:
    - Root note and chord quality
    - Inversion and voicing
    - Extensions (7th, 9th, 11th, 13th)
    - Tension and consonance
    """

    domain_tag = DOMAIN_TAGS["music_chord"]
    domain_name = "music_chord"

    def __init__(self):
        self._quality_codes = {
            "major": 0,
            "minor": 1,
            "diminished": 2,
            "augmented": 3,
            "major 7": 4,
            "minor 7": 5,
            "dominant 7": 6,
            "half-diminished 7": 7,
            "fully-diminished 7": 8,
        }

    def encode(
        self,
        root: Union[int, str],
        quality: str = "major",
        inversion: int = 0,
        extensions: Optional[List[int]] = None,
    ) -> Any:
        """
        Encode a musical chord.

        Args:
            root: Root note (MIDI or string)
            quality: Chord quality ('major', 'minor', etc.)
            inversion: Inversion number (0, 1, 2, or 3)
            extensions: List of extension semitones (7, 9, 11, 13)

        Returns:
            128-dim embedding
        """
        # Parse root to MIDI
        if isinstance(root, int):
            root_midi = root
        else:
            root_midi = PitchEncoder()._parse_note_string(root)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Root note
        emb = backend.at_add(emb, 8 + CHORD_ROOT_OFFSET, root_midi / 127.0)

        # Chord quality
        quality_code = self._quality_codes.get(quality, 0) / 10.0
        emb = backend.at_add(emb, 8 + CHORD_QUALITY_OFFSET, quality_code)

        # Inversion
        emb = backend.at_add(emb, 8 + CHORD_INVERSION, inversion / 4.0)

        # Extensions
        if extensions is None:
            extensions = []

        for i, ext in enumerate(extensions[:4]):
            emb = backend.at_add(emb, 8 + CHORD_EXTENSION_OFFSET + i, ext / 14.0)

        # Interval pattern (semitones from root)
        pattern = CHORD_PATTERNS.get(quality, [0, 4, 7])
        for i, interval in enumerate(pattern):
            emb = backend.at_add(emb, 8 + CHORD_INTERVAL_OFFSET + i, interval / 12.0)

        # Tension score (simple heuristic)
        tension = self._calculate_tension(pattern, extensions)
        emb = backend.at_add(emb, 8 + CHORD_TENSION_OFFSET, tension)

        return emb

    def decode(self, emb: Any) -> Tuple[int, str, int]:
        """
        Decode embedding to (root_midi, quality, inversion).
        """
        root_midi = int(round(emb[8 + CHORD_ROOT_OFFSET].item() * 127.0))
        quality_code = int(round(emb[8 + CHORD_QUALITY_OFFSET].item() * 10.0))
        inversion = int(round(emb[8 + CHORD_INVERSION].item() * 4.0))

        # Reverse lookup quality
        quality = {v: k for k, v in self._quality_codes.items()}.get(
            quality_code, "major"
        )

        return root_midi, quality, inversion

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid chord."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Chord operations (deterministic)
    # ========================================================================

    def transpose(self, emb: Any, semitones: int) -> Any:
        """Transpose a chord by semitone interval."""
        root_midi, quality, inversion = self.decode(emb)
        new_root = root_midi + semitones
        return self.encode(new_root, quality, inversion)

    def interval_between(self, emb1: Any, emb2: Any) -> int:
        """Calculate interval between chord roots."""
        root1 = int(round(emb1[8 + CHORD_ROOT_OFFSET].item() * 127.0))
        root2 = int(round(emb2[8 + CHORD_ROOT_OFFSET].item() * 127.0))
        return root2 - root1

    def is_consonant(self, emb: Any) -> bool:
        """
        Check if chord is consonant.

        Based on interval content.
        """
        tension = emb[8 + CHORD_TENSION_OFFSET].item()
        return tension < 0.5  # Simple threshold

    def has_common_tones(self, emb1: Any, emb2: Any) -> bool:
        """Check if two chords have common tones."""
        root1 = int(round(emb1[8 + CHORD_ROOT_OFFSET].item() * 127.0))
        root2 = int(round(emb2[8 + CHORD_ROOT_OFFSET].item() * 127.0))

        # Get chord tones (simplified)
        quality1 = int(round(emb1[8 + CHORD_QUALITY_OFFSET].item() * 10.0))
        quality2 = int(round(emb2[8 + CHORD_QUALITY_OFFSET].item() * 10.0))

        pattern1 = CHORD_PATTERNS.get(
            {v: k for k, v in self._quality_codes.items()}.get(quality1, "major"),
            [0, 4, 7],
        )
        pattern2 = CHORD_PATTERNS.get(
            {v: k for k, v in self._quality_codes.items()}.get(quality2, "major"),
            [0, 4, 7],
        )

        tones1 = {(root1 + i) % 12 for i in pattern1}
        tones2 = {(root2 + i) % 12 for i in pattern2}

        return len(tones1 & tones2) > 0

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_tension(self, pattern: List[int], extensions: List[int]) -> float:
        """Calculate chord tension score (0-1)."""
        # Simple heuristic: count dissonant intervals
        dissonant_intervals = {1, 6, 10, 11}  # Minor 2nd, tritone, minor 7th, major 7th

        tension = 0.0
        for interval in pattern:
            if interval in dissonant_intervals:
                tension += 0.2

        for ext in extensions:
            if ext in dissonant_intervals:
                tension += 0.3

        return min(tension, 1.0)


# =============================================================================
# Scale Encoding (Melodic Structure)
# =============================================================================

# Embedding layout for SCALE (dims 8-71):
# dims 0-1:   Root note MIDI number (normalized)
# dim 2:      Scale type (encoded as integer)
# dims 3-10:  Scale degrees (which chromatic notes are in scale)
# dims 11-18: Scale degree intervals (semitones from root)
# dims 19-24: Mode characteristics (bright/dark, etc.)
# dims 25-32: Key signature
# dims 33-47: Common chords in key
# dims 48-63: Reserved for modulation

SCALE_ROOT_OFFSET = 0
SCALE_TYPE_OFFSET = 2
SCALE_DEGREES_OFFSET = 3
SCALE_INTERVALS_OFFSET = 11
SCALE_MODE_OFFSET = 19
SCALE_KEY_SIG_OFFSET = 25
SCALE_CHORDS_OFFSET = 33


class ScaleEncoder:
    """
    Encoder for musical scales and modes.

    Captures:
    - Root note and scale type
    - Scale degrees and intervals
    - Mode characteristics (bright/dark)
    - Key signature
    - Diatonic chords
    """

    domain_tag = DOMAIN_TAGS["music_scale"]
    domain_name = "music_scale"

    def __init__(self):
        self._scale_codes = {
            "major": 0,
            "natural minor": 1,
            "harmonic minor": 2,
            "melodic minor": 3,
            "pentatonic major": 4,
            "pentatonic minor": 5,
            "blues": 6,
            "dorian": 7,
            "phrygian": 8,
            "lydian": 9,
            "mixolydian": 10,
            "locrian": 11,
        }

    def encode(self, root: Union[int, str], scale_type: str = "major") -> Any:
        """
        Encode a musical scale.

        Args:
            root: Root note (MIDI or string)
            scale_type: Scale type ('major', 'natural minor', etc.)

        Returns:
            128-dim embedding
        """
        backend = get_backend()
        # Parse root to MIDI
        if isinstance(root, int):
            root_midi = root
        else:
            root_midi = PitchEncoder()._parse_note_string(root)

        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Root note
        emb = backend.at_add(emb, 8 + SCALE_ROOT_OFFSET, root_midi / 127.0)

        # Scale type
        scale_code = self._scale_codes.get(scale_type, 0) / 12.0
        emb = backend.at_add(emb, 8 + SCALE_TYPE_OFFSET, scale_code)

        # Scale degrees (which chromatic notes are in scale)
        pattern = SCALE_PATTERNS.get(scale_type, [0, 2, 4, 5, 7, 9, 11])
        for i in range(12):
            if i in pattern:
                emb = backend.at_add(emb, 8 + SCALE_DEGREES_OFFSET + i, 1.0)
            else:
                emb = backend.at_add(emb, 8 + SCALE_DEGREES_OFFSET + i, 0.0)

        # Scale intervals (semitones from root)
        for i, interval in enumerate(pattern):
            emb = backend.at_add(emb, 8 + SCALE_INTERVALS_OFFSET + i, interval / 12.0)

        return emb

    def decode(self, emb: Any) -> Tuple[int, str]:
        """
        Decode embedding to (root_midi, scale_type).
        """
        root_midi = int(round(emb[8 + SCALE_ROOT_OFFSET].item() * 127.0))
        scale_code = int(round(emb[8 + SCALE_TYPE_OFFSET].item() * 12.0))

        # Reverse lookup scale type
        scale_type = {v: k for k, v in self._scale_codes.items()}.get(
            scale_code, "major"
        )

        return root_midi, scale_type

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is a valid scale."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Scale operations (deterministic)
    # ========================================================================

    def relative_minor(self, emb: Any) -> Any:
        """Get relative minor of a major scale."""
        root_midi, scale_type = self.decode(emb)
        if scale_type == "major":
            new_root = root_midi - 3
            return self.encode(new_root, "natural minor")
        return emb  # Already minor

    def parallel_minor(self, emb: Any) -> Any:
        """Get parallel minor (same root, different scale)."""
        root_midi, scale_type = self.decode(emb)
        if scale_type == "major":
            return self.encode(root_midi, "natural minor")
        elif scale_type == "natural minor":
            return self.encode(root_midi, "major")
        return emb

    def contains_note(self, emb: Any, note_midi: int) -> bool:
        """Check if a note is in the scale."""
        root_midi = int(round(emb[8 + SCALE_ROOT_OFFSET].item() * 127.0))
        chromatic_degree = (note_midi - root_midi) % 12

        # Check if this degree is in the scale
        degree_flag = emb[8 + SCALE_DEGREES_OFFSET + chromatic_degree].item()
        return degree_flag > 0.5

    def modulation_distance(self, emb1: Any, emb2: Any) -> int:
        """Calculate modulation distance (number of common tones)."""
        common = 0
        for i in range(12):
            deg1 = emb1[8 + SCALE_DEGREES_OFFSET + i].item()
            deg2 = emb2[8 + SCALE_DEGREES_OFFSET + i].item()
            if deg1 > 0.5 and deg2 > 0.5:
                common += 1

        return common


# =============================================================================
# Utility Functions
# =============================================================================


def get_interval_name(semitones: int) -> str:
    """Get interval name from semitone count."""
    interval_map = {
        0: "unison",
        1: "minor second",
        2: "major second",
        3: "minor third",
        4: "major third",
        5: "fourth",
        6: "tritone",
        7: "fifth",
        8: "minor sixth",
        9: "major sixth",
        10: "minor seventh",
        11: "major seventh",
        12: "octave",
    }
    return interval_map.get(semitones % 12, "unknown")


def note_to_freq(note: str) -> float:
    """Convert note string to frequency."""
    encoder = PitchEncoder()
    emb = encoder.encode(note)
    log_freq = emb[8 + PITCH_FREQ_OFFSET].item()
    return math.exp(log_freq)


def freq_to_note(freq: float) -> str:
    """Convert frequency to nearest note."""
    midi = freq_to_midi(freq)
    note, octave = midi_to_note_octave(round(midi))
    return f"{note}{octave}"


# =============================================================================
# Atonal Theory - 12-Tone Serialism
# =============================================================================

# Embedding layout for PC_SET (dims 8-71):
# dims 0-11:  Pitch class set (12-dimensional binary vector)
# dims 12-23: Interval class vector (6 intervals, each counted twice)
# dims 24-35: Prime form representation
# dim 36:     Cardinality (number of elements)
# dim 37:     Fortean number (decimal encoding)
# dims 38-47: Invariance vector (under Tn/I operations)
# dims 48-63: Fortean number matrix representation

PC_SET_VECTOR_OFFSET = 0
IC_VECTOR_OFFSET = 12
PRIME_FORM_OFFSET = 24
CARDINALITY_OFFSET = 36
FORTEAN_OFFSET = 37
INVARIANCE_OFFSET = 38
FORTEAN_MATRIX_OFFSET = 48


def pitch_class_set_to_vector(pcs: List[int]) -> Any:
    """Convert pitch class set to a 12-dimensional indicator vector.

    Notes
    -----
    Example: {0, 3, 7, 11} -> [1,0,0,1,0,0,0,1,0,0,0,1]
    """
    backend = get_backend()
    vec = backend.zeros(12)
    for pc in pcs:
        pc_mod = pc % 12
        vec = backend.at_add(vec, pc_mod, 1.0 - vec[pc_mod])
    return vec


def normal_form(pcs: List[int]) -> List[int]:
    """
    Compute normal form of a pitch class set.

    Normal form = most compact left-packed rotation.

    Notes
    -----
    Follows the normal-form heuristic described by Forte (1973).
    """
    if not pcs:
        return []

    # Generate all rotations
    rotations = []
    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for i in range(12):
        rotated = [((pc - i) % 12) for pc in sorted_pcs]
        rotations.append(rotated)

    # Find most compact (minimizes span between first and last)
    def span(rotation):
        if len(rotation) == 1:
            return 0
        return (rotation[-1] - rotation[0]) % 12

    min_span = min(span(r) for r in rotations)
    candidates = [r for r in rotations if span(r) == min_span]

    # Choose leftmost (lexicographically smallest)
    return min(candidates)


def prime_form(pcs: List[int]) -> List[int]:
    """
    Compute prime form (most compact form + its inverse).

    Prime form = normal form or its inverse, whichever is most compact.

    """
    nf = normal_form(pcs)

    # Inversion (11 - n for each n)
    inv = sorted([((11 - pc) % 12) for pc in pcs])
    inv_nf = normal_form(inv)

    # Compare spans
    def span(rotation):
        if len(rotation) == 1:
            return 0
        return (rotation[-1] - rotation[0]) % 12

    if span(nf) <= span(inv_nf):
        return nf
    else:
        return inv_nf


def interval_class_vector(pcs: List[int]) -> Any:
    """
    Compute interval class vector (ICV).

    ICV = counts of interval classes 1-6.

    Example: {0, 4, 7} (C, E, G - major triad)
    → ICV = [0, 0, 1, 0, 1, 1] (major 3rd, P5, minor 6th)

    """
    backend = get_backend()
    icv = backend.zeros(6)

    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for i, pc1 in enumerate(sorted_pcs):
        for pc2 in sorted_pcs[i + 1 :]:
            interval = (pc2 - pc1) % 12
            ic_class = min(interval, 12 - interval)  # Interval class 1-6

            if ic_class > 0 and ic_class <= 6:
                idx = ic_class - 1
                icv = backend.at_add(icv, idx, 1.0)

    return icv


def forte_number_helper(pcs: List[int]) -> int:
    """
    Compute Fortean number (decimal encoding of pitch class set).

    Forte number = binary digits of PC set interpreted as decimal.

    Example: {0, 3, 7, 11} = 100100010001₂ = 2337

    """
    vec = pitch_class_set_to_vector(pcs)
    forte = 0

    for i in range(12):
        if vec[i].item() > 0.5:
            forte += 2**i

    return forte


def transposition(pcs: List[int], n: int) -> List[int]:
    """
    Transpose pitch class set by n semitones (Tn operation).

    Notes
    -----
    Applies ``(pc + n) mod 12`` to each element.
    """
    return [(pc + n) % 12 for pc in pcs]


def inversion(pcs: List[int]) -> List[int]:
    """
    Invert pitch class set (I operation).

    Notes
    -----
    Applies ``(11 - pc) mod 12`` to each element.
    """
    return [((11 - pc) % 12) for pc in pcs]


def tn_operation(pcs: List[int], n: int) -> List[int]:
    """
    Tn operation (transposition by n semitones).
    """
    return transposition(pcs, n)


def tin_operation(pcs: List[int], n: int) -> List[int]:
    """
    TnI operation (invert then transpose by n).
    """
    inv = inversion(pcs)
    return transposition(inv, n)


def interval_class_similarity(icv1: Any, icv2: Any) -> float:
    """
    Compute ISIM (Interval Class SIMilarity).

    ISIM = dot product of IC vectors, normalized.

    """
    backend = get_backend()
    dot = backend.sum(icv1 * icv2).item()
    norm1 = backend.sqrt(backend.sum(icv1**2)).item()
    norm2 = backend.sqrt(backend.sum(icv2**2)).item()

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


def asim(icv1: Any, icv2: Any) -> float:
    """
    Compute ASIM (Absolute SIMilarity).

    ASIM = 6 - sum(|ICV1 - ICV2|)

    """
    backend = get_backend()
    diff = backend.abs(icv1 - icv2)
    sum_abs = backend.sum(diff).item()
    return 6 - sum_abs


def invariant_under_Tn(pcs: List[int]) -> List[int]:
    """
    Find all transposition invariants (Tn where Tn(S) = S).

    """
    invariants = []

    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for n in range(12):
        transposed = sorted([(pc + n) % 12 for pc in sorted_pcs])
        if transposed == sorted_pcs:
            invariants.append(n)

    return invariants


def invariant_under_TnI(pcs: List[int]) -> List[int]:
    """
    Find all TnI invariants (TnI(S) = S).

    """
    invariants = []

    sorted_pcs = sorted(set(pc % 12 for pc in pcs))

    for n in range(12):
        inverted = sorted([((11 - pc) % 12) for pc in sorted_pcs])
        transposed = sorted([(pc + n) % 12 for pc in inverted])

        if transposed == sorted_pcs:
            invariants.append(n)

    return invariants


def z_related(pcs1: List[int], pcs2: List[int]) -> bool:
    """
    Check if two sets are Z-related (same ICV, different sets).

    Z-relation = same interval class content, but different notes.

    """
    backend = get_backend()
    icv1 = interval_class_vector(pcs1)
    icv2 = interval_class_vector(pcs2)

    icv_equal = bool(backend.allclose(icv1, icv2, atol=0.01).item())

    set1 = sorted(set(pc % 12 for pc in pcs1))
    set2 = sorted(set(pc % 12 for pc in pcs2))

    sets_different = set1 != set2

    return icv_equal and sets_different


def subset_of(pcs1: List[int], pcs2: List[int]) -> bool:
    """
    Check if pcs1 is a subset of pcs2.

    """
    set1 = set(pc % 12 for pc in pcs1)
    set2 = set(pc % 12 for pc in pcs2)
    return set1.issubset(set2)


def complement(pcs: List[int]) -> List[int]:
    """
    Compute complement (12-tone scale minus the set).

    """
    universe = set(range(12))
    pcs_set = set(pc % 12 for pc in pcs)
    complement_set = universe - pcs_set
    return sorted(complement_set)


def multiplication(pcs: List[int], n: int) -> List[int]:
    """
    M-n operation (multiplication by n mod 12).

    Used for serial music (12-tone rows).

    """
    return [((pc * n) % 12) for pc in pcs]


def row_matrix_helper(row: List[int]) -> Any:
    """
    Create row matrix (12x12) for 12-tone row.

    Each row is Tn(I(row)) operation.

    """
    backend = get_backend()
    matrix = backend.zeros((12, 12))

    for n in range(12):
        transformed = tin_operation(row, n)
        for i, pc in enumerate(transformed):
            matrix = backend.at_add(matrix, n, i, float(pc))

    return matrix


def matrix_multiply_row(matrix: Any, row: Any) -> Any:
    """
    Multiply row by atonal matrix (serial operation).

    """
    backend = get_backend()
    result = backend.zeros(12)

    for i in range(12):
        val = 0.0
        for j in range(12):
            val += matrix[i, j].item() * row[j].item()
        result = backend.at_add(result, i, val - result[i])

    return result


class AtonalSetEncoder:
    """
    Encoder for atonal pitch class sets.

    Embeds:
    - Pitch class set (12-dim binary vector)
    - Interval class vector (6-dim counts)
    - Prime form (normalized representation)
    - Fortean number (decimal encoding)
    - Invariance properties
    """

    domain_tag = DOMAIN_TAGS["music_atonal"]
    domain_name = "music_atonal"

    def encode(self, pcs: List[int]) -> Any:
        """
        Encode a pitch class set.

        Args:
            pcs: List of pitch classes (0-11), e.g., [0, 4, 7]

        Returns:
            128-dim embedding with full atonal analysis
        """
        backend = get_backend()
        emb = create_embedding()

        # Domain tag
        emb = backend.at_add(emb, slice(0, 8), self.domain_tag)

        # Pitch class set (12-dim binary vector)
        pc_vec = pitch_class_set_to_vector(pcs)
        for i in range(12):
            emb = backend.at_add(emb, 8 + PC_SET_VECTOR_OFFSET + i, pc_vec[i])

        # Interval class vector (6-dim)
        icv = interval_class_vector(pcs)
        for i in range(6):
            emb = backend.at_add(emb, 8 + IC_VECTOR_OFFSET + i, icv[i])

        # Prime form
        pf = prime_form(pcs)
        for i, pc in enumerate(pf):
            emb = backend.at_add(emb, 8 + PRIME_FORM_OFFSET + i, float(pc) / 12.0)

        # Cardinality
        emb = backend.at_add(emb, 8 + CARDINALITY_OFFSET, 
            float(len(set(pc % 12 for pc in pcs))) / 12.0
        )

        # Fortean number
        forte = forte_number_helper(pcs)
        _, log_forte = log_encode_value(float(forte))
        emb = backend.at_add(emb, 8 + FORTEAN_OFFSET, log_forte / 12.0)

        # Tn invariance (which transpositions preserve set)
        tn_inv = invariant_under_Tn(pcs)
        for n in tn_inv:
            emb = backend.at_add(emb, 8 + INVARIANCE_OFFSET + n, 1.0)

        return emb

    def decode(self, emb: Any) -> List[int]:
        """
        Decode embedding back to pitch class set.
        """
        # Extract pitch class set
        pcs = []
        for i in range(12):
            if emb[8 + PC_SET_VECTOR_OFFSET + i].item() > 0.5:
                pcs.append(i)

        return pcs

    def is_valid(self, emb: Any) -> bool:
        """Check if embedding is valid."""
        backend = get_backend()
        tag = emb[0:8]
        return bool(backend.allclose(tag, self.domain_tag, atol=0.1).item())

    # ========================================================================
    # Atonal operations (matrix operations)
    # ========================================================================

    def Tn(self, emb: Any, n: int) -> Any:
        """
        Transpose by n semitones.
        """
        pcs = self.decode(emb)
        transposed = transposition(pcs, n)
        return self.encode(transposed)

    def I(self, emb: Any) -> Any:
        """
        Invert (I operation).
        """
        pcs = self.decode(emb)
        inverted = inversion(pcs)
        return self.encode(inverted)

    def TnI(self, emb: Any, n: int) -> Any:
        """
        Invert then transpose by n (TnI operation).
        """
        pcs = self.decode(emb)
        transformed = tin_operation(pcs, n)
        return self.encode(transformed)

    def M_n(self, emb: Any, n: int) -> Any:
        """
        Multiply by n (M-n operation).
        """
        pcs = self.decode(emb)
        multiplied = multiplication(pcs, n)
        return self.encode(multiplied)

    def similarity(self, emb1: Any, emb2: Any) -> float:
        """
        Compute ISIM (interval class similarity).
        """
        pcs1 = self.decode(emb1)
        pcs2 = self.decode(emb2)

        icv1 = interval_class_vector(pcs1)
        icv2 = interval_class_vector(pcs2)

        return interval_class_similarity(icv1, icv2)

    def is_z_related(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if two sets are Z-related.
        """
        pcs1 = self.decode(emb1)
        pcs2 = self.decode(emb2)
        return z_related(pcs1, pcs2)

    def is_subset(self, emb1: Any, emb2: Any) -> bool:
        """
        Check if emb1 is subset of emb2.
        """
        pcs1 = self.decode(emb1)
        pcs2 = self.decode(emb2)
        return subset_of(pcs1, pcs2)

    def get_prime_form(self, emb: Any) -> Any:
        """
        Get prime form embedding.
        """
        backend = get_backend()
        pcs = self.decode(emb)
        pf = prime_form(pcs)

        result = self.encode(pcs)

        # Override with prime form only
        for i, pc in enumerate(pf):
            result = backend.at_add(result, 8 + PRIME_FORM_OFFSET + i, 
                float(pc) / 12.0 - result[8 + PRIME_FORM_OFFSET + i]
            )

        return result

    def is_invariant_under_Tn(self, emb: Any, n: int) -> bool:
        """
        Check if set is invariant under Tn.
        """
        pcs = self.decode(emb)
        invariants = invariant_under_Tn(pcs)
        return n in invariants

    def get_icv(self, emb: Any) -> Any:
        """
        Extract interval class vector.
        """
        pcs = self.decode(emb)
        return interval_class_vector(pcs)

    def create_row_matrix(self, emb: Any) -> Any:
        """
        Create 12x12 row matrix for serial composition.
        """
        pcs = self.decode(emb)
        return row_matrix(pcs)

    def get_icv(self, emb: Any) -> Any:
        """
        Extract interval class vector.
        """
        pcs = self.decode(emb)
        return interval_class_vector(pcs)

    def create_row_matrix(self, emb: Any) -> Any:
        """
        Create 12x12 row matrix for serial composition.
        """
        pcs = self.decode(emb)
        return row_matrix(pcs)
