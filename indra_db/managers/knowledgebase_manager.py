import os
import tempfile
from collections import defaultdict

import boto3
import pickle
import logging

from indra_db.util import insert_db_stmts
from indra_db.util.distill_statements import extract_duplicates, KeyFunc

logger = logging.getLogger(__name__)


class KnowledgebaseManager(object):
    """This is a class to lay out the methods for updating a dataset."""
    name = NotImplemented

    def upload(self, db):
        """Upload the content for this dataset into the database."""
        dbid = self._check_reference(db)
        stmts = self._get_statements()
        insert_db_stmts(db, stmts, dbid)
        return

    def update(self, db):
        """Add any new statements that may have come into the dataset."""
        dbid = self._check_reference(db, can_create=False)
        if dbid is None:
            raise ValueError("This knowledge base has not yet been "
                             "registered.")
        existing_keys = set(db.select_all([db.RawStatements.mk_hash,
                                           db.RawStatements.source_hash],
                                          db.RawStatements.db_info_id == dbid))
        stmts = self._get_statements()
        filtered_stmts = [s for s in stmts
                          if (s.get_hash(), s.evidence[0].get_source_hash())
                          not in existing_keys]
        insert_db_stmts(db, filtered_stmts, dbid)
        return

    def _check_reference(self, db, can_create=True):
        """Ensure that this database has an entry in the database."""
        dbid = db.select_one(db.DBInfo.id, db.DBInfo.db_name == self.name)
        if dbid is None:
            if can_create:
                dbid = db.insert(db.DBInfo, db_name=self.name)
            else:
                return None
        else:
            dbid = dbid[0]
        return dbid

    def _get_statements(self):
        raise NotImplementedError("Statement retrieval must be defined in "
                                  "each child.")


class TasManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of the TAS dataset."""
    name = 'tas'

    @staticmethod
    def get_order_value(stmt):
        cm = stmt.evidence[0].annotations['class_min']
        return ['Kd < 100nM', '100nM < Kd < 1uM'].index(cm)

    def choose_better(self, *stmts):
        best_stmt = min([(self.get_order_value(stmt), stmt)
                         for stmt in stmts if stmt is not None],
                        key=lambda t: t[0])
        return best_stmt[1]

    def _get_statements(self):
        from indra.sources.tas import process_csv
        proc = process_csv()
        stmts, dups = extract_duplicates(proc.statements)
        print(dups)
        return stmts


class SignorManager(KnowledgebaseManager):
    name = 'signor'

    def _get_statements(self):
        from indra.sources.signor import process_from_web
        proc = process_from_web()
        return proc.statements


class CBNManager(KnowledgebaseManager):
    """This manager handles retrieval and processing of CBN network files"""
    name = 'cbn'

    def __init__(self, archive_url=None):
        if not archive_url:
            self.archive_url = ('http://www.causalbionet.com/Content'
                                '/jgf_bulk_files/Human-2.0.zip')
        else:
            self.archive_url = archive_url
        return

    def _get_statements(self):
        import requests
        from zipfile import ZipFile
        from indra.sources.bel.api import process_cbn_jgif_file
        import tempfile

        cbn_dir = tempfile.mkdtemp('cbn_manager')

        logger.info('Retrieving CBN network zip archive')
        tmp_zip = os.path.join(cbn_dir, 'cbn_human.zip')
        resp = requests.get(self.archive_url)
        with open(tmp_zip, 'wb') as f:
            f.write(resp.content)

        stmts = []
        tmp_dir = os.path.join(cbn_dir, 'cbn')
        os.mkdir(tmp_dir)
        with ZipFile(tmp_zip) as zipf:
            logger.info('Extracting archive to %s' % tmp_dir)
            zipf.extractall(path=tmp_dir)
            logger.info('Processing jgif files')
            for jgif in zipf.namelist():
                if jgif.endswith('.jgf') or jgif.endswith('.jgif'):
                    logger.info('Processing %s' % jgif)
                    pbp = process_cbn_jgif_file(os.path.join(tmp_dir, jgif))
                    stmts += pbp.statements

        uniques, dups = extract_duplicates(stmts,
                                           key_func=KeyFunc.mk_and_one_ev_src)

        logger.info("Deduplicating...")
        print('\n'.join(str(dup) for dup in dups))
        print(len(dups))

        return uniques


class BiogridManager(KnowledgebaseManager):
    name = 'biogrid'

    def _get_statements(self):
        from indra.sources import biogrid
        bp = biogrid.BiogridProcessor()
        return list(_expanded(bp.statements))


class PathwayCommonsManager(KnowledgebaseManager):
    name = 'pc10'
    skips = {'psp', 'hprd'}

    def __init__(self, *args, **kwargs):
        self.counts = defaultdict(lambda: 0)
        super(PathwayCommonsManager, self).__init__(*args, **kwargs)

    def _can_include(self, stmt):
        num_ev = len(stmt.evidence)
        assert num_ev == 1, "Found statement with %d evidence." % num_ev

        ev = stmt.evidence[0]
        ssid = ev.annotations['source_sub_id']
        self.counts[ssid] += 1

        return ssid not in self.skips

    def _get_statements(self):
        s3 = boto3.client('s3')

        resp = s3.get_object(Bucket='bioexp-paper',
                             Key='bioexp_biopax_pc10.pkl')
        stmts = pickle.loads(resp['Body'].read())

        filtered_stmts = [s for s in _expanded(stmts) if self._can_include(s)]
        return filtered_stmts


class HPRDManager(KnowledgebaseManager):
    name = 'hprd'

    def _get_statements(self):
        import tarfile
        import requests
        from indra.sources import hprd

        # Download the files.
        hprd_base = 'http://www.hprd.org/RELEASE9/'
        resp = requests.get(hprd_base + 'HPRD_FLAT_FILES_041310.tar.gz')
        tmp_dir = tempfile.mkdtemp('hprd_files')
        tmp_tarfile = os.path.join(tmp_dir, 'hprd_files.tar.gz')
        with open(tmp_tarfile, 'wb') as f:
            f.write(resp.content)

        # Extract the files.
        with tarfile.open(tmp_tarfile, 'r:gz') as tf:
            tf.extractall(tmp_dir)

        # Find the relevant files.
        dirs = os.listdir(tmp_dir)
        for files_dir in dirs:
            if files_dir.startswith('FLAT_FILES'):
                break
        files_path = os.path.join(tmp_dir, files_dir)
        file_names = {'id_mappings_file': 'HPRD_ID_MAPPINGS',
                      'complexes_file': 'PROTEIN_COMPLEXES',
                      'ptm_file': 'POST_TRANSLATIONAL_MODIFICATIONS',
                      'ppi_file': 'BINARY_PROTEIN_PROTEIN_INTERACTIONS',
                      'seq_file': 'PROTEIN_SEQUENCES'}
        kwargs = {kw: os.path.join(files_path, fname + '.txt')
                  for kw, fname in file_names.items()}

        # Run the processor
        hp = hprd.process_flat_files(**kwargs)

        # Filter out exact duplicates
        unique_stmts, dups = \
            extract_duplicates(_expanded(hp.statements),
                               key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))

        return unique_stmts


class BelLcManager(KnowledgebaseManager):
    name = 'bel_lc'

    def _get_statements(self):
        from indra.sources import bel

        s3 = boto3.client('s3')
        resp = s3.get_object(Bucket='bigmech', Key='indra-db/large_corpus.bel')
        tmp_bel = tempfile.mktemp('lc.bel')
        with open(tmp_bel, 'wb') as f:
            f.write(resp['Body'].read())
        pbp = bel.process_belscript(tmp_bel)
        stmts, dups = extract_duplicates(pbp.statements,
                                         key_func=KeyFunc.mk_and_one_ev_src)
        print('\n'.join(str(dup) for dup in dups))
        print(len(stmts), len(dups))
        return stmts


def _expanded(stmts):
    for stmt in stmts:
        # Only one evidence is allowed for each statement.
        if len(stmt.evidence) > 1:
            for ev in stmt.evidence:
                new_stmt = stmt.make_generic_copy()
                new_stmt.evidence.append(ev)
                yield new_stmt
        else:
            yield stmt