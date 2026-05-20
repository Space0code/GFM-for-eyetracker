# Diploma v LaTeXu

Ta mapa vsebuje lokalno verzijo diplomske naloge, pripravljeno iz FRI LaTeX predloge.

Glavna datoteka je:

```bash
diploma/diploma.tex
```

Lokalni prevod iz terminala:

```bash
cd diploma
latexmk diploma.tex
```

Vmesne datoteke in PDF se zapišejo v `gradnja/`.

Struktura:

- `uvodni_del/`: povzetek, abstract in seznam kratic.
- `poglavja/`: glavna poglavja diplome.
- `dodatki/`: dodatki na koncu naloge.
- `slike/`: slike, namenjene diplomi.
- `tabele/`: tabele, namenjene diplomi.
- `podporno/`: podporne datoteke FRI predloge.
