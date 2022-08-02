"""
This is an example module for using genotype-query engines for gemini.
An engine only has to have a query() function that returns a list of
variant ids that meet a gemini genotype query.

Any engine can be added in a post-hoc fashion provided a module is
provided with the single filter():

    # take a gemini --gt-filter string and return a list of variant_ids that meet
    # that filter. `obj` is the thing returned by load().
    # user_dict contains things like HET, UNKNOWN, etc. used in the eval.
    filter(db_path, obj, gt_filter, user_dict)

See below for an implementation using bcolz.
It is using carray rather than ctable because with ctable, we'd be limited to
2908 samples on a ext3 file-system. with carrays, we can have up to 31998
samples on ext3. These limits are not an issue for ext4.

"""
from __future__ import absolute_import

import os
import sys
sys.setrecursionlimit(8192)
import time
import re
import shutil
import zlib

import numpy as np
import bcolz
import numexpr as ne
bcolz.blosc_set_nthreads(2)
ne.set_num_threads(2)
import sqlalchemy as sql
from . import database

from . import compression
from .gemini_utils import get_gt_cols

def get_samples(metadata):
    return [x['name'] for x in metadata.tables['samples'].select().order_by("sample_id").execute()]

def get_n_variants(cur):
    return next(iter(cur.execute(sql.text("select count(*) from variants"))))[0]

def get_bcolz_dir(db):
    if "://" not in db:
        return f"{db}.gts"
    base = os.environ.get("gemini_bcolz_path", os.path.expand("~/.bcolz/"))
    return os.path.join(base, db.split("/")[-1])

gt_cols_types = (
    ('gts', np.object),
    ('gt_types', np.int8),
    ('gt_phases', np.bool),
    ('gt_depths', np.int32),
    ('gt_ref_depths', np.int32),
    ('gt_alt_depths', np.int32),
    ('gt_alt_freqs', np.float32),
    ('gt_quals', np.float32),
    ('gt_copy_numbers', np.int32),
    ('gt_phred_ll_homref', np.int32),
    ('gt_phred_ll_het', np.int32),
    ('gt_phred_ll_homalt', np.int32),
)

def mkdir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass

def create(db, cols=None):
    if cols is None:
        cols = [x[0] for x in gt_cols_types if x[0] != 'gts']
        sys.stderr.write(
                "indexing all columns except 'gts'; to index that column, "
                "run gemini bcolz_index %s --cols gts\n" % db)

    conn, metadata = database.get_session_metadata(db)
    gt_cols = [x for x in get_gt_cols(metadata) if x in cols]
    samples = get_samples(metadata)
    bcpath = get_bcolz_dir(db)

    mkdir(bcpath)

    nv = get_n_variants(conn)

    sys.stderr.write("loading %i variants for %i samples into bcolz\n"
                     % (nv, len(samples)))

    if nv == 0 or len(samples) == 0:
        return

    carrays = {}
    tmps = {}
    try:
        for gtc in gt_cols:
            carrays[gtc] = []
            tmps[gtc] = []

            dt = dict(gt_cols_types)[gtc]
            for s in samples:
                mkdir(f"{bcpath}/{s}")
                carrays[gtc].append(
                    bcolz.carray(
                        np.empty(0, dtype=dt),
                        expectedlen=nv,
                        rootdir=f"{bcpath}/{s}/{gtc}",
                        chunklen=16384 * 8,
                        mode="w",
                    )
                )

                tmps[gtc].append([])

        t0 = time.time()
        # scale step by number of samples to limit memory use.
        step = max(100, 2000000 / len(samples))
        sys.stderr.write("step-size: %i\n" % step)
        del gtc
        decomp = compression.unpack_genotype_blob

        empty = [-1] * len(samples)
        for i, row in enumerate(conn.execute(sql.text(f'select {", ".join(gt_cols)} from variants'))):
            if i == 0:
                try:
                    decomp(row[0])
                except zlib.error:
                    decomp = compression.snappy_unpack_blob

            for j, gt_col in enumerate(gt_cols):
                vals = decomp(row[j])
                if vals is None or len(vals) == 0:  # empty gt_phred_ll
                    vals = empty
                for isamp, sample in enumerate(samples):
                    tmps[gt_col][isamp].append(vals[isamp])
                    if (i > 0 and i % step == 0) or i == nv - 1:
                        carrays[gt_col][isamp].append(tmps[gt_col][isamp])
                        tmps[gt_col][isamp] = []
                        carrays[gt_col][isamp].flush()

            if i % step == 0 and i > 0:
                sys.stderr.write("at %.1fM (%.0f rows / second)\n" % (i / 1000000., i / float(time.time() - t0)))

        t = float(time.time() - t0)
        sys.stderr.write("loaded %d variants at %.1f / second\n" %
                         (len(carrays[gt_col][0]), nv / t))
    except:
        # on error, we remove the dirs so we can't have weird problems.
        for k, li in carrays.items():
            for i, ca in enumerate(li):
                if i < 5:
                    sys.stderr("removing: %s\n" % ca.rootdir)
                if i == 5:
                    sys.stderr("not reporting further removals for %s\n" % k)
                ca.flush()
                shutil.rmtree(ca.rootdir)
        raise


class NoGTIndexException(Exception):
    pass

# TODO: since we call this from query, we can improve speed by only loading
# samples that appear in the query with an optional query=None arg to load.
def load(db, query=None):

    t0 = time.time()
    conn, metadata = database.get_session_metadata(db)

    gt_cols = get_gt_cols(metadata)
    samples = get_samples(metadata)
    bcpath = get_bcolz_dir(db)

    carrays = {}
    n = 0
    for gtc in gt_cols:
        if gtc not in query: continue
        carrays[gtc] = []
        for s in samples:
            if s not in query and fix_sample_name(s) not in query:
                # need to add anyway as place-holder
                carrays[gtc].append(None)
                continue
            path = f"{bcpath}/{s}/{gtc}"
            if os.path.exists(path):
                carrays[gtc].append(bcolz.open(path, mode="r"))
                n += 1
    if os.environ.get("GEMINI_DEBUG") == "TRUE":
        sys.stderr.write("it took %.2f seconds to load %d arrays\n" \
            % (time.time() - t0, n))
    return carrays

def fix_sample_name(s):
    return s.replace("-", "_").replace(" ", "_")

def filter(db, query, user_dict):
    # these should be translated to a bunch or or/and statements within gemini
    # so they are supported, but must be translated before getting here.
    if query == "False" or query is None or query is False:
        return []
    if "any(" in query or "all(" in query or \
       ("sum(" in query and not query.startswith("sum(") and query.count("sum(") == 1):
        return None
    user_dict['where'] = np.where

    if query.startswith("not "):
        # "~" is not to numexpr.
        query = f"~{query[4:]}"
    sum_cmp = False
    if query.startswith("sum("):
        assert query[-1].isdigit()
        query, sum_cmp = query[4:].rsplit(")", 1)
        query = f"({query}) {sum_cmp}"

    query = query.replace(".", "__")
    query = " & ".join(f"({token})" for token in query.split(" and "))
    query = " | ".join(f"({token})" for token in query.split(" or "))

    conn, metadata = database.get_session_metadata(db)
    samples = get_samples(metadata)
    # convert gt_col[index] to gt_col__sample_name
    patt = "(%s)\[(\d+)\]" % "|".join((g[0] for g in gt_cols_types))


    def subfn(x):
        """Turn gt_types[1] into gt_types__sample"""
        field, idx = x.groups()
        return f"{field}__{fix_sample_name(samples[int(idx)])}"

    query = re.sub(patt, subfn, query)
    if os.environ.get('GEMINI_DEBUG') == 'TRUE':
        sys.stderr.write(query[:250] + "...\n")
    carrays = load(db, query=query)

    if len(carrays) == 0 or max(len(carrays[c]) for c in carrays) == 0 or \
       any(not any(carrays[c]) for c in carrays):
       # need this 2nd check above because of the place-holders in load()
        raise NoGTIndexException

    # loop through and create a cache of "$gt__$sample"
    for gt_col in carrays:
        if gt_col not in query: continue
        for i, sample_array in enumerate(carrays[gt_col]):
            sample = fix_sample_name(samples[i])
            if sample not in query: continue
            user_dict[f"{gt_col}__{sample}"] = sample_array

    # had to special-case count. it won't quite be as efficient
    if "|count|" in query:
        tokens = query[2:-2].split("|count|")
        icmp = tokens[-1]
        # a list of carrays, so not in memory.
        res = [bcolz.eval(tok, user_dict=user_dict) for tok in tokens[:-1]]
        # in memory after this, but just a single axis array.
        res = np.sum(res, axis=0)
        res = ne.evaluate(f'res{icmp}')
    else:
        res = bcolz.eval(query, user_dict=user_dict)

    try:
        if res.shape[0] == 1 and len(res.shape) > 1:
            res = res[0]
    except AttributeError:
        return []
    variant_ids, = np.where(res)
    #variant_ids = np.array(list(bcolz.eval(query, user_dict=user_dict,
    #    vm="numexpr").wheretrue()))
    # variant ids are 1-based.
    return 1 + variant_ids if len(variant_ids) > 0 else []

if __name__ == "__main__":

    db = sys.argv[1]
    #create(sys.argv[1])
    carrays = load(db)
    conn, metadata = database.get_session_metadata(db)
    if len(sys.argv) > 2:
        q = sys.argv[2]
    else:
        q = "gt_types.1094PC0012 == HET and gt_types.1719PC0016 == HET and gts.1094PC0012 == 'A/C'"

    print(filter(db, carrays, q, user_dict=dict(HET=1, HOM_REF=0, HOM_ALT=3,
        UNKNOWN=2)))
    print("compare to:", ("""gemini query -q "select variant_id, gts.1719PC0016 from variants" """
                          """ --gt-filter "%s" %s""" % (q, db)))

