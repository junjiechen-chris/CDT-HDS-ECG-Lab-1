"""Task 1 of Lab 1: six excerpts from N-labelled recordings, each with ten consecutive wave points
(the P, Q, R, S and T of two beats, starting at a different wave each time) marked as dots. The
positions were taken from NeuroKit2's ecg_delineate (method "peak") and checked by eye, then
stored here so that the task does not depend on the library version. The first diagram is shown
with its letters as the example; the other five are the task. The expected letters are stored
as salted hashes so that the check can say which dots are wrong without printing the answer."""
import hashlib

import matplotlib.pyplot as plt
import numpy as np

from shared_ecg import FS, signal_mv

_SALT = "lab1-waves"
EXAMPLE_LABELS = "PQRSTPQRST"
DIAGRAMS = [
    {"rid": "A01702", "start": 1417, "stop": 1839,
     "dots": [1465, 1479, 1500, 1510, 1568, 1673, 1689, 1709, 1719, 1782],
     "hashes": ['48efd87ace73', 'a26c585c3f1c', '3f595af740af', '6711c5ffa2fa', 'b616f2db53b1', 'b783520cfc7b', '83dcf76c4df8', '6883715cdf34', '25030f16b685', '06d164ff7116']},
    {"rid": "A00419", "start": 533, "stop": 1075,
     "dots": [540, 565, 574, 642, 791, 806, 832, 842, 906, 1067],
     "hashes": ['66e467da9d1a', '1a4c5431dfc0', '9073014dc30b', 'f91ab0934a25', 'a6d3cb2f13c4', '50526ae22850', 'bad52ac1c913', 'c8f2db556e0d', 'aec10e023a85', '913648490816']},
    {"rid": "A00261", "start": 1029, "stop": 1572,
     "dots": [1037, 1047, 1119, 1261, 1288, 1308, 1318, 1389, 1538, 1560],
     "hashes": ['bd7c476525d6', '1fb56cd403e3', '1650fe640896', '9a88c5f51341', 'a44c265589f0', '247c8e2768d6', '5fc267091558', '80d317af55a2', '438c3b3318da', 'a80c611f638e']},
    {"rid": "A01483", "start": 1333, "stop": 1882,
     "dots": [1338, 1421, 1559, 1592, 1603, 1613, 1695, 1837, 1866, 1877],
     "hashes": ['705e38f25d9a', 'f50fba7d647a', '847ba5ca0732', '58f6726c6288', '456ff50a0b0a', 'a31106ed8d5e', '38cb4daf5311', '43c76259344e', '859defb3c274', 'b7952dfac6aa']},
    {"rid": "A00105", "start": 455, "stop": 986,
     "dots": [494, 631, 656, 666, 677, 755, 900, 925, 936, 947],
     "hashes": ['a29302d7cb7d', '078665bb5dba', '918d0e26da49', '1f4782ccf121', '8ea90616e3e6', '949227a648db', '0dc32338aacd', 'bc0394b2f7cc', '1d130b3265e7', '2b7322c35c79']},
    {"rid": "A00799", "start": 6422, "stop": 6842,
     "dots": [6476, 6498, 6510, 6533, 6578, 6686, 6710, 6722, 6740, 6789],
     "hashes": ['344c0f06985b', 'c4ff320f995e', '01a41f046141', 'fb99ae080174', 'f41eeaaa04fc', '4453eecebc62', 'e4d5abbd71df', 'd860e94cc19f', '93b55da61697', '26476ac46448']},
]


def _hash(rid, n, letter):
    return hashlib.sha1(f"{_SALT}:{rid}:{n}:{letter}".encode()).hexdigest()[:12]


def show_wave_diagrams(ecg_data):
    """Plot the example with its letters, and diagrams 1-5 with numbered dots."""
    fig, axes = plt.subplots(len(DIAGRAMS), 1, figsize=(12, 2.4 * len(DIAGRAMS)))
    for n, (ax, d) in enumerate(zip(axes, DIAGRAMS)):
        x = signal_mv(ecg_data[d["rid"]])
        t = np.arange(len(x)) / FS
        a, b = d["start"], d["stop"]
        ax.plot(t[a:b], x[a:b], lw=0.9, color="C0")
        for k, i in enumerate(d["dots"]):
            ax.plot(i / FS, x[i], "o", color="C1", ms=6)
            text = EXAMPLE_LABELS[k] if n == 0 else str(k + 1)
            ax.annotate(text, (i / FS, x[i]), (0, 9), textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color="C1" if n == 0 else "black")
        title = "Example, labelled" if n == 0 else f"Diagram {n}: label the {len(d['dots'])} dots in order"
        ax.set_title(f"{title}  (recording {d['rid']}, label N)", loc="left", fontsize=10)
        ax.set_ylabel("ECG (mV)"); ax.grid(True, alpha=0.4)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(); plt.show()


def check_wave_labels(answers):
    """answers: {1: "Q R S T P Q R S T P", ...}. Prints which dots are wrong, never the answer."""
    all_ok = True
    for n in range(1, len(DIAGRAMS)):
        d = DIAGRAMS[n]
        given = answers.get(n)
        if not given:
            print(f"Diagram {n}: no answer yet."); all_ok = False; continue
        letters = [c for c in str(given).upper() if c in "PQRST"]
        if len(letters) != len(d["dots"]):
            print(f"Diagram {n}: {len(d['dots'])} dots but {len(letters)} letters given."); all_ok = False; continue
        wrong = [k + 1 for k, l in enumerate(letters) if _hash(d["rid"], k + 1, l) != d["hashes"][k]]
        if wrong:
            print(f"Diagram {n}: dots {wrong} are wrong."); all_ok = False
        else:
            print(f"Diagram {n}: correct.")
    if all_ok:
        print("Task 1 check passed: all five diagrams labelled correctly.")
    return all_ok
