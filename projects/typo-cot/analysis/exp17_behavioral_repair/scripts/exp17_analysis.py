#!/usr/bin/env python3
"""Exp17 behavioral-repair analysis (self-contained, GPU-free, read-only inputs).

Detects self-correction markers in DeepSeek-R1-Distill-Qwen-7B CoT (<think>) text
and cross-tabulates against flip and R_Q (token importance_score).

Markers (operational definitions):
  markA  correct spelling of a perturbed token appears in CoT (implicit read-through)
  markT  the typo (perturbed) form is reproduced verbatim in CoT
  markC  markA AND markT for the same token (explicit typo-and-correction co-occurrence)
  cue    an explicit typo/wording-awareness phrase matches CORRECTION_RE anywhere in CoT
  repair_explicit (sample-level) = cue OR (any token markC)

Run:  python3 exp17_analysis.py   # writes ../raw_output.txt
"""
import json, re, math, statistics, random, os

WT10 = "/diskthalys/ssd14tc/sfukuhata/dev/kanolab/typo_robust_analysis/.claude/worktrees/exp-10-scope/projects/typo-cot"
BASE = WT10 + "/outputs/baseline/%s_%s/results.json"
PERT = WT10 + "/outputs/perturbed/%s_%s_k4_%s/results.json"
M = "DeepSeek-R1-Distill-Qwen-7B"
HERE = os.path.dirname(os.path.abspath(__file__))

STOPWORDS = set("""a an the of to in on at for and or but is are was were be been being
this that these those it its he she they them his her their you your we our i me my
as by with from into over under than then so if not no yes do does did has have had
which who whom whose what when where why how all any some each every both few more most
other such only own same too very can will just should now here there out up down off
about above below between through during before after because while""".split())

CORRECTION_CUES = [
    r"typo", r"misspell", r"mispell", r"mis-spell", r"spelled", r"spelling",
    r"meant to (say|write|type|be)", r"supposed to (say|read|be|mean)",
    r"assuming (they|you|the question|the problem|it|this) (mean|meant|means)",
    r"i (think|believe|assume|guess) (they|the question|the problem|this|it|you) (mean|meant|means)",
    r"(they|the question|the problem|author) (probably|likely|must have|might have|may have) mean",
    r"presumably (mean|means|meant|they|the)", r"which i (assume|think|believe|guess)",
    r"the word ['\"]?\w", r"there'?s? (a|an) typo", r"seems? (like|to be) a typo",
    r"i think (this|that|it) (is|was) a typo", r"correct (spelling|word) (is|would)",
]
CORRECTION_RE = re.compile("|".join("(?:%s)" % c for c in CORRECTION_CUES), re.IGNORECASE)
STRICT_RE = re.compile(r"typo|misspell|mispell|meant to (say|write|type)|supposed to (say|read)|seems? (like|to be) a typo", re.IGNORECASE)


def norm(t): return (t or "").strip()
def is_content_word(o):
    o = norm(o); return bool(re.fullmatch(r"[A-Za-z]+", o)) and len(o) >= 4 and o.lower() not in STOPWORDS
def present(form, cotl):
    f = norm(form)
    if not f: return False
    fl = f.lower()
    if re.fullmatch(r"[A-Za-z]+", f): return re.search(r"\b" + re.escape(fl) + r"\b", cotl) is not None
    return fl in cotl
def load(p): return json.load(open(p))
def rate(c, n): return c / n if n else float("nan")

def odds_ratio(a, b, c, d):
    if 0 in (a, b, c, d): a, b, c, d = a + .5, b + .5, c + .5, d + .5
    o = (a * d) / (b * c); se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    return o, math.exp(math.log(o) - 1.96*se), math.exp(math.log(o) + 1.96*se)
def two_prop_z(x1, n1, x2, n2):
    if n1 == 0 or n2 == 0: return float("nan"), float("nan")
    p = (x1 + x2) / (n1 + n2); se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    if se == 0: return float("nan"), float("nan")
    z = (x1/n1 - x2/n2) / se; return z, math.erfc(abs(z)/math.sqrt(2))
def qbins(vals, nb=5):
    xs = sorted(vals); cuts = [xs[min(len(xs)-1, int(round(q*len(xs))))] for q in [i/nb for i in range(1, nb)]]
    def which(v):
        for i, c in enumerate(cuts):
            if v <= c: return i
        return nb - 1
    return cuts, which

def build(base_path, pert_path, cond, task):
    base = {s["sample_id"]: bool(s["is_correct"]) for s in load(base_path)}
    recs, srecs = [], []
    for s in load(pert_path):
        sid = s["sample_id"]; cot = s.get("cot_text") or ""; cotl = cot.lower()
        bc = base.get(sid); pc = bool(s["is_correct"]); flip = (bc is True) and (not pc)
        cue = CORRECTION_RE.search(cot) is not None
        anyC = anyAct = False
        for t in (s.get("perturbed_tokens") or []):
            orig = norm(t.get("original_token", "")); typo = norm(t.get("perturbed_token", ""))
            mA = present(orig, cotl); mT = present(typo, cotl); mC = mA and mT; ct = is_content_word(orig)
            if mC: anyC = True
            if mA and ct: anyAct = True
            recs.append(dict(task=task, cond=cond, sample_id=sid, R_Q=t.get("importance_score"),
                             ptype=t.get("perturbation_type"), content=ct, markA=mA, markT=mT, markC=mC,
                             base_correct=bc, flip=flip))
        srecs.append(dict(task=task, cond=cond, sample_id=sid, base_correct=bc, pert_correct=pc, flip=flip,
                          cue=cue, any_markC=anyC, any_markA_content=anyAct, repair_explicit=(cue or anyC)))
    return recs, srecs

TOK, SAMP = {}, {}
for task in ["math", "gsm8k", "mmlu"]:
    for cond in ["importance", "random"]:
        TOK[(task, cond)], SAMP[(task, cond)] = build(BASE % (M, task), PERT % (M, task, cond), cond, task)

out = []
def p(*a):
    line = " ".join(str(x) for x in a); out.append(line); print(line)

p("="*92); p("EXP17 BEHAVIORAL REPAIR - " + M); p("="*92)

p("\n## 0. Flip rates (among baseline-correct samples). Reversal = math random>importance.")
p("%-8s %-12s %6s %6s %8s %8s %8s" % ("task", "cond", "nSamp", "nBC", "pertAcc", "flipN", "flipRate"))
for task in ["math", "gsm8k", "mmlu"]:
    for cond in ["importance", "random"]:
        sr = SAMP[(task, cond)]; nBC = sum(1 for s in sr if s["base_correct"]); fl = sum(1 for s in sr if s["flip"])
        p("%-8s %-12s %6d %6d %8.3f %8d %8.3f" % (task, cond, len(sr), nBC,
          rate(sum(1 for s in sr if s["pert_correct"]), len(sr)), fl, rate(fl, nBC)))

p("\n## 1. Sample-level marker prevalence")
p("%-8s %-12s %6s %8s %8s %8s %8s" % ("task", "cond", "nSamp", "cue", "anyC", "anyA_ct", "explicit"))
for task in ["math", "gsm8k", "mmlu"]:
    for cond in ["importance", "random"]:
        sr = SAMP[(task, cond)]; n = len(sr)
        p("%-8s %-12s %6d %8.3f %8.3f %8.3f %8.3f" % (task, cond, n,
          rate(sum(s["cue"] for s in sr), n), rate(sum(s["any_markC"] for s in sr), n),
          rate(sum(s["any_markA_content"] for s in sr), n), rate(sum(s["repair_explicit"] for s in sr), n)))
p("\n   Importance vs Random within task (two-prop z):")
for label, key in [("explicit", "repair_explicit"), ("cue", "cue")]:
    for task in ["math", "gsm8k", "mmlu"]:
        si = SAMP[(task, "importance")]; sr = SAMP[(task, "random")]
        xi = sum(s[key] for s in si); xr = sum(s[key] for s in sr); z, pv = two_prop_z(xi, len(si), xr, len(sr))
        p("     %-9s %-6s imp=%.3f rand=%.3f z=%.2f p=%.3f" % (label, task, rate(xi, len(si)), rate(xr, len(sr)), z, pv))

p("\n## 2. repair x flip cross-tab (baseline-correct), OR of flip given repair")
for cuelabel, RE, useC in [("broad(cue|markC)", CORRECTION_RE, True), ("strict-cue-only", STRICT_RE, False)]:
    p("   --- %s ---" % cuelabel)
    p("   %-8s %-10s %7s %8s %8s %9s %s" % ("task", "cond", "R&flip", "R&noflp", "NR&flip", "NR&noflp", "stats"))
    for task in ["math", "gsm8k", "mmlu"]:
        base = {s["sample_id"]: bool(s["is_correct"]) for s in load(BASE % (M, task))}
        cells = {"importance": [0, 0, 0, 0], "random": [0, 0, 0, 0], "BOTH": [0, 0, 0, 0]}
        for cond in ["importance", "random"]:
            for s in load(PERT % (M, task, cond)):
                if not base.get(s["sample_id"]): continue
                cot = s.get("cot_text") or ""; cotl = cot.lower(); flip = not bool(s["is_correct"])
                R = RE.search(cot) is not None
                if useC and not R:
                    for t in (s.get("perturbed_tokens") or []):
                        if present(norm(t.get("original_token", "")), cotl) and present(norm(t.get("perturbed_token", "")), cotl):
                            R = True; break
                idx = (0 if flip else 1) if R else (2 if flip else 3)
                cells[cond][idx] += 1; cells["BOTH"][idx] += 1
        for cond in ["importance", "random", "BOTH"]:
            a, b, c, d = cells[cond]; o, lo, hi = odds_ratio(a, b, c, d)
            p("   %-8s %-10s %7d %8d %8d %9d  f|R=%.3f f|NR=%.3f OR=%.2f[%.2f,%.2f]" %
              (task, cond, a, b, c, d, rate(a, a+b), rate(c, c+d), o, lo, hi))

p("\n## 3. R_Q-quintile x marker rate (token-level, pooled importance+random)")
p("   markA=correct-form ; markC=typo&correct co-occur ; markA(ct)=content-word subset")
for task in ["math", "gsm8k", "mmlu"]:
    toks = [t for t in TOK[(task, "importance")] + TOK[(task, "random")] if t["R_Q"] is not None]
    cuts, which = qbins([t["R_Q"] for t in toks], 5)
    p("   -- %s  cuts=%s" % (task, ["%.4f" % c for c in cuts]))
    p("      %-4s %6s %10s %8s %8s %9s" % ("Q", "nTok", "R_Q_med", "markA", "markC", "markA(ct)"))
    for q in range(5):
        tq = [t for t in toks if which(t["R_Q"]) == q]; ctq = [t for t in tq if t["content"]]
        p("      Q%-3d %6d %10.4f %8.3f %8.3f %9.3f" % (q, len(tq), statistics.median([t["R_Q"] for t in tq]),
          rate(sum(t["markA"] for t in tq), len(tq)), rate(sum(t["markC"] for t in tq), len(tq)),
          rate(sum(t["markA"] for t in ctq), len(ctq)) if ctq else float("nan")))

p("\n## 3b. R_Q-quintile x marker, CONTENT WORDS ONLY")
for task in ["math", "gsm8k", "mmlu"]:
    toks = [t for t in TOK[(task, "importance")] + TOK[(task, "random")] if t["R_Q"] is not None and t["content"]]
    cuts, which = qbins([t["R_Q"] for t in toks], 5)
    p("   -- %s  nContentTok=%d  cuts=%s" % (task, len(toks), ["%.4f" % c for c in cuts]))
    p("      %-4s %6s %10s %8s %8s" % ("Q", "nTok", "R_Q_med", "markA", "markC"))
    for q in range(5):
        tq = [t for t in toks if which(t["R_Q"]) == q]
        p("      Q%-3d %6d %10.4f %8.3f %8.3f" % (q, len(tq), statistics.median([t["R_Q"] for t in tq]),
          rate(sum(t["markA"] for t in tq), len(tq)), rate(sum(t["markC"] for t in tq), len(tq))))

p("\n## 4. Implicit read-through markA_frac by condition (baseline-correct)")
p("   %-8s %-12s %6s %12s %14s %12s" % ("task", "cond", "nBC", "frac_flip", "frac_noflip", "frac_all"))
for task in ["math", "gsm8k", "mmlu"]:
    for cond in ["importance", "random"]:
        recs = TOK[(task, cond)]; bytok = {}
        for r in recs: bytok.setdefault(r["sample_id"], []).append(r)
        ff, nf, allf = [], [], []
        for s in SAMP[(task, cond)]:
            if not s["base_correct"]: continue
            ts = bytok.get(s["sample_id"], [])
            if not ts: continue
            fr = sum(t["markA"] for t in ts) / len(ts); allf.append(fr)
            (ff if s["flip"] else nf).append(fr)
        p("   %-8s %-12s %6d %12.3f %14.3f %12.3f" % (task, cond, len(allf),
          statistics.mean(ff) if ff else float("nan"), statistics.mean(nf) if nf else float("nan"),
          statistics.mean(allf) if allf else float("nan")))

p("\n## 4c. markT (typo form reproduced) token rate by condition")
for task in ["math", "gsm8k", "mmlu"]:
    for cond in ["importance", "random"]:
        recs = TOK[(task, cond)]
        p("   %-8s %-12s nTok=%5d markT=%.3f" % (task, cond, len(recs), rate(sum(t["markT"] for t in recs), len(recs))))

p("\n## 5. FP AUDIT: 20 random MATH CoTs matching broad cue (both conds)")
random.seed(17); pool = []
for cond in ["importance", "random"]:
    for s in load(PERT % (M, "math", cond)):
        cot = s.get("cot_text") or ""; m = CORRECTION_RE.search(cot)
        if m: pool.append((cond, s["sample_id"], m, cot))
random.shuffle(pool)
for cond, sid, m, cot in pool[:20]:
    st = max(0, m.start()-70); en = min(len(cot), m.end()+70)
    p("   [%s %s] cue='%s' | ...%s..." % (cond, sid, m.group(0), cot[st:en].replace("\n", " ")))

open(os.path.join(HERE, "..", "raw_output.txt"), "w").write("\n".join(out))
print("\n[written ../raw_output.txt]")
