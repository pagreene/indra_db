import json
import pickle
import logging
from os import path, remove, listdir, makedirs
from functools import wraps
from datetime import datetime
from collections import defaultdict
from argparse import ArgumentParser

from indra.util import batch_iter, clockit
from indra.statements import Statement
from indra.tools import assemble_corpus as ac
from indra.preassembler.sitemapper import logger as site_logger
from indra.preassembler.grounding_mapper.mapper import logger \
                                                    as grounding_logger
from indra.preassembler import Preassembler
from indra.preassembler import logger as ipa_logger
from indra.preassembler.hierarchy_manager import hierarchies

from indra_db.util import insert_pa_stmts, distill_stmts, get_db, \
    extract_agent_data, insert_pa_agents

site_logger.setLevel(logging.INFO)
grounding_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

HERE = path.dirname(path.abspath(__file__))
ipa_logger.setLevel(logging.INFO)


def _preassembly_wrapper(meth):
    @wraps(meth)
    def wrap_preassembly(pam, db, *args, **kwargs):
        pam._register_preassembly_start(meth.__name__, *args, **kwargs)
        try:
            completed = meth(pam, db, *args, **kwargs)
        except Exception as e:
            # Pickle the entire manager for debugging.
            logger.exception(e)
            pam.fossilize(e)
            completed = False
        pam._register_preassembly_end(db, completed)
        return completed
    return wrap_preassembly


def _tag(tag):

    def tag_wrapper(meth):

        @wraps(meth)
        def tagged_method(pam, *args, **kwargs):
            tag_set = pam._set_tag(tag)
            res = meth(pam, *args, **kwargs)
            if tag_set:
                pam._unset_tag()
            return res

        return tagged_method
    return tag_wrapper


class IndraDBPreassemblyError(Exception):
    pass


DATE_FMT = '%Y%m%d_%H%M%S'


class PreassemblyManager(object):
    """Class used to manage the preassembly pipeline

    Parameters
    ----------
    n_proc : int
        Select the number of processes that will be used when performing
        preassembly. Default is 1.
    batch_size : int
        Select the maximum number of statements you wish to be handled at a
        time. In general, a larger batch size will somewhat be faster, but
        require much more memory.
    """
    def __init__(self, n_proc=1, batch_size=10000, print_logs=False):
        self.n_proc = n_proc
        self.batch_size = batch_size
        self.pa = Preassembler(hierarchies)
        self.__tag = 'Unpurposed'
        self.__print_logs = print_logs

        # Variables set during preassembly.
        self.__original_start = None
        self.__pa_start = None
        self.__pa_variant = None
        self.__cache = None
        self.__continuing = None

        # Variablues used during preassembly
        self.raw_sids = None
        self.stmts = None
        self.cleaned_stmts = None
        self.new_pa_stmts = None
        self.new_agent_tuples = None
        self.new_evidence_links = None
        self.uuid_sid_dict = None
        self.mk_done = None
        self.mk_new = None
        return

    def _set_tag(self, tag):
        if self.__tag == 'Unpurposed':
            self.__tag = tag
            return True
        return False

    def _unset_tag(self):
        self.__tag = 'Unpurposed'

    def fossilize(self, error=None):
        pkl_file = 'manager_fossil_%s.pkl' % self.__pa_start.strftime(DATE_FMT)

        # Convert all defaultdicts to dicts for pickling.
        for name, val in self.__dict__.items():
            if isinstance(val, defaultdict):
                setattr(self, name, dict(val))

        with open(path.join(self.__cache, pkl_file), 'wb') as f:
            pickle.dump({'preassembly_manager': self, 'error': error}, f)

        return

    def _register_preassembly_start(self, method_name, continuing=False):
        self.__pa_start = datetime.utcnow()
        self.__pa_variant = method_name
        self.__continuing = continuing
        if continuing:
            superdir = path.join(HERE, '.%s_caches' % method_name)
            subdir = max(listdir(superdir),
                         key=lambda name: datetime.strptime(name, DATE_FMT))
            self.__cache = path.join(superdir, subdir)
            self.__original_start = datetime.strptime(subdir, DATE_FMT)
        else:
            self.__original_start = self.__pa_start
            # Create the cache
            self.__cache = path.join(HERE, '.%s_caches' % method_name,
                                     self.__pa_start.strftime(DATE_FMT))
            makedirs(self.__cache)

        return

    def _register_preassembly_end(self, db, completed):
        if completed:
            is_corpus_init = (self.__pa_variant == 'create_corpus')
            db.insert('preassembly_updates', corpus_init=is_corpus_init,
                      run_datetime=self.__pa_start)

        self.__pa_start = None
        self.__pa_variant = None
        self.__original_start = None
        self.__cache = None
        self.__continuing = None
        pass

    def _get_latest_updatetime(self, db):
        """Get the date of the latest update."""
        update_list = db.select_all(db.PreassemblyUpdates)
        if not len(update_list):
            logger.warning("The preassembled corpus has not been initialized, "
                           "or else the updates table has not been populated.")
            return None
        return max([u.run_datetime for u in update_list])

    def _run_cached(self, func, args=None, kwargs=None, other_data=None,
                    before_comps=None, after_comps=None):
        # Handle None other_data
        if other_data is None:
            other_data = {}

        # Define the path to the pickle.
        pkl_path = path.join(self.__cache, func.__name__ + "_results.pkl")

        # If we are continuing, just reload the content from the cache.
        if self.__continuing and path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self._log("Loading %s results from %s."
                          % (func.__name__, pkl_path))
                pkl_dict = pickle.load(f)
                if pkl_dict['other_data']:
                    return pkl_dict['ret'], pkl_dict['other_data']
                return pkl_dict['ret']

        # Set arg and kwargs defaults.
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        # Run pre-function computations
        if before_comps:
            for label, comp in before_comps.items():
                other_data[label] = comp()

        # Run the function.
        ret = func(*args, **kwargs)

        # Run post-function computations
        if after_comps:
            for label, comp in after_comps.items():
                other_data[label] = comp()

        # Dump the results to the cache.
        with open(pkl_path, 'wb') as f:
            self._log("Dumping results of %s to %s."
                      % (func.__name__, pkl_path))
            if other_data:
                pickle.dump({'ret': ret, 'other_data': other_data}, f)
            else:
                pickle.dump({'ret': ret, 'other_data': None}, f)

        return ret

    def _raw_sid_stmt_iter(self, db, do_enumerate=False):
        """Return a generator over statements with the given database ids."""
        def _fixed_raw_stmt_from_json(s_json, tr):
            stmt = _stmt_from_json(s_json)
            if tr is not None:
                stmt.evidence[0].pmid = tr.pmid
                stmt.evidence[0].text_refs = {k: v
                                              for k, v in tr.__dict__.items()
                                              if not k.startswith('_')}
            return stmt

        i = 0
        for stmt_id_batch in batch_iter(self.raw_sids, self.batch_size):
            subres = (db.filter_query([db.RawStatements.id,
                                      db.RawStatements.json,
                                      db.TextRef],
                                     db.RawStatements.id.in_(stmt_id_batch))
                        .outerjoin(db.Reading)
                        .outerjoin(db.TextContent)
                        .outerjoin(db.TextRef)
                        .yield_per(self.batch_size//10))
            data = [(sid, _fixed_raw_stmt_from_json(s_json, tr))
                    for sid, s_json, tr in subres]
            if do_enumerate:
                yield i, data
                i += 1
            else:
                yield data

    @clockit
    def _extract_and_push_unique_statements(self, db):
        """Get the unique Statements from the raw statements."""
        self._log("There are %d distilled raw statement ids to preassemble."
                  % len(self.raw_sids))

        self.mk_new = self._get_cached_set('mk_new')

        num_batches = len(self.raw_sids) / self.batch_size
        for i, stmt_tpl_batch in self._raw_sid_stmt_iter(db, True):
            try:
                self._log("Processing batch %d/%d of %d/%d statements."
                          % (i, num_batches, len(stmt_tpl_batch),
                             len(self.raw_sids)))
                # Get list of statements, generate mapping from uuid to sid.
                self.stmts = []
                self.uuid_sid_dict = {}
                for sid, stmt in stmt_tpl_batch:
                    self.uuid_sid_dict[stmt.uuid] = sid
                    self.stmts.append(stmt)

                # Map groundings and sequences.
                self._clean_statements()

                # Use the shallow hash to condense unique statements.
                self._condense_statements()

                # Insert the statements and their links.
                self._log("Insert new statements into database...")
                insert_pa_stmts(db, self.new_pa_stmts, ignore_agents=True,
                                commit=False)
                self._log("Insert new raw_unique links into the database...")
                db.copy('raw_unique_links', self._flattened_evidence_dict(),
                        ('pa_stmt_mk_hash', 'raw_stmt_id'), commit=False)
                db.copy('pa_agents', self.new_agent_tuples,
                        ('stmt_mk_hash', 'ag_num', 'db_name', 'db_id', 'role'),
                        lazy=True, commit=False)
                insert_pa_agents(db, self.new_pa_stmts, verbose=True,
                                 skip=['agents'])  # This will commit

                # Reset the pa statements
                self.new_pa_stmts = None
                self.new_agent_tuples = None
                self.new_evidence_links = None
            finally:
                self.mk_done.dump()
                self.mk_new.dump()

        self._log("Added %d new pa statements into the database."
                  % len(self.mk_new))
        return

    def _flattened_evidence_dict(self):
        return {(u_stmt_key, ev_stmt_uuid)
                for u_stmt_key, ev_stmt_uuid_set in self.new_evidence_links.items()
                for ev_stmt_uuid in ev_stmt_uuid_set}

    @clockit
    def _condense_statements(self):
        self._log("Condense into unique statements...")
        self.new_pa_stmts = []
        self.new_evidence_links = defaultdict(lambda: set())
        self.new_agent_tuples = set()
        for s in self.cleaned_stmts:
            h = shash(s, refresh=True)

            # If this statement is new, make it.
            if h not in self.mk_done and h not in self.mk_new:
                self.new_pa_stmts.append(s.make_generic_copy())
                self.mk_new.add(h)

            # Add the evidence to the dict.
            self.new_evidence_links[h].add(self.uuid_sid_dict[s.uuid])

            # Add any db refs to the agents.
            ref_data, _, _ = extract_agent_data(s, h)
            self.new_agent_tuples |= set(ref_data)

        return

    @_preassembly_wrapper
    @_tag('create')
    def create_corpus(self, db, continuing=False):
        """Initialize the table of preassembled statements.

        This method will find the set of unique knowledge represented in the
        table of raw statements, and it will populate the table of preassembled
        statements (PAStatements/pa_statements), while maintaining links between
        the raw statements and their unique (pa) counterparts. Furthermore, the
        refinement/support relationships between unique statements will be found
        and recorded in the PASupportLinks/pa_support_links table.

        For more detail on preassembly, see indra/preassembler/__init__.py
        """
        # Get filtered statement ID's.
        self.raw_sids = self._run_cached(distill_stmts, args=[db])

        # Handle the possibility we're picking up after an earlier job...
        self.mk_done = self._get_cached_set('mk_done')
        if continuing:
            self._log("Getting set of statements already de-duplicated...")
            link_resp = db.select_all([db.RawUniqueLinks.raw_stmt_id,
                                       db.RawUniqueLinks.pa_stmt_mk_hash])
            if link_resp:
                checked_raw_stmt_ids, pa_stmt_hashes = \
                    zip(*db.select_all([db.RawUniqueLinks.raw_stmt_id,
                                        db.RawUniqueLinks.pa_stmt_mk_hash]))
                self.raw_sids -= set(checked_raw_stmt_ids)
                self.mk_done |= self._get_cached_set('mk_done', pa_stmt_hashes)
                self._log("Found %d preassembled statements already done."
                          % len(self.mk_done))

        # Get the set of unique statements
        self._extract_and_push_unique_statements(db)

        # If we are continuing, check for support links that were already found
        if continuing:
            self._log("Getting pre-existing links...")
            db_existing_links = db.select_all([
                db.PASupportLinks.supporting_mk_hash,
                db.PASupportLinks.supporting_mk_hash
                ])
            existing_links = {tuple(res) for res in db_existing_links}
            self._log("Found %d existing links." % len(existing_links))
        else:
            existing_links = set()

        # Now get the support links between all batches.
        support_links = set()
        outer_iter = db.select_all_batched(self.batch_size,
                                           db.PAStatements.json,
                                           order_by=db.PAStatements.mk_hash)
        for outer_idx, outer_batch_jsons in outer_iter:
            outer_batch = [_stmt_from_json(sj) for sj, in outer_batch_jsons]
            # Get internal support links
            self._log('Getting internal support links outer batch %d.'
                      % outer_idx)
            some_support_links = self._get_support_links(outer_batch)

            # Get links with all other batches
            inner_iter = db.select_all_batched(self.batch_size,
                                               db.PAStatements.json,
                                               order_by=db.PAStatements.mk_hash,
                                               skip_idx=outer_idx)
            for inner_idx, inner_batch_jsons in inner_iter:
                inner_batch = [_stmt_from_json(sj) for sj, in inner_batch_jsons]
                split_idx = len(inner_batch)
                full_list = inner_batch + outer_batch
                self._log('Getting support between outer batch %d and inner'
                          'batch %d.' % (outer_idx, inner_idx))
                some_support_links |= \
                    self._get_support_links(full_list, split_idx=split_idx)

            # Add all the new support links
            support_links |= (some_support_links - existing_links)

            # There are generally few support links compared to the number of
            # statements, so it doesn't make sense to copy every time, but for
            # long preassembly, this allows for better failure recovery.
            if len(support_links) >= self.batch_size:
                self._log("Copying batch of %d support links into db."
                          % len(support_links))
                db.copy('pa_support_links', support_links,
                        ('supported_mk_hash', 'supporting_mk_hash'))
                existing_links |= support_links
                support_links = set()

        # Insert any remaining support links.
        if support_links:
            self._log("Copying final batch of %d support links into db."
                      % len(support_links))
            db.copy('pa_support_links', support_links,
                    ('supported_mk_hash', 'supporting_mk_hash'))

        return True

    def _get_new_stmt_ids(self, db):
        """Get all the uuids of statements not included in evidence."""
        old_id_q = db.filter_query(
            db.RawStatements.id,
            db.RawStatements.id == db.RawUniqueLinks.raw_stmt_id
        )
        new_sid_q = db.filter_query(db.RawStatements.id).except_(old_id_q)
        all_new_stmt_ids = {sid for sid, in new_sid_q.all()}
        self._log("Found %d new statement ids." % len(all_new_stmt_ids))
        return all_new_stmt_ids

    @_tag('supplement')
    def _supplement_statements(self, db):
        """Supplement the preassembled statements with the latest content."""
        last_update = self._get_latest_updatetime(db)
        start_date = datetime.utcnow()
        self._log("Latest update was: %s" % last_update)

        # Get the new statements...
        self._log("Loading info about the existing state of preassembly. "
                  "(This may take a little time)")
        new_ids = self._run_cached(self._get_new_stmt_ids, args=[db])

        # Weed out exact duplicates.
        stmt_ids = self._run_cached(distill_stmts, args=[db],
                                    kwargs=dict(get_full_stmts=False))
        self.raw_sids = new_ids & stmt_ids

        # Get the set of new unique statements and link to any new evidence.
        self.mk_done = {mk for mk, in db.select_all(db.PAStatements.mk_hash)}
        self._log("Found %d old pa statements." % len(self.mk_done))

        new_mk_set, time_data = \
            self._run_cached(self._extract_and_push_unique_statements,
                             args=[db],
                             other_data={'start_date': start_date},
                             after_comps={'end_date': datetime.utcnow})
        start_date = time_data['start_date']
        end_date = time_data['end_date']

        if self.__continuing:
            self._log("Original old mk set: %d" % len(self.mk_done))
            old_mk_set = self.mk_done - new_mk_set
            self._log("Adjusted old mk set: %d" % len(old_mk_set))

        self._log("Found %d new pa statements." % len(new_mk_set))
        return start_date, end_date

    def _get_cached_set(self, name, iterable=None):
        pkl_path = path.join(self.__cache, name + "_set_cache.pkl")

        if self.__continuing:
            s = CachedSet.load(pkl_path)
            if iterable:
                s |= set(iterable)
        if iterable:
            return CachedSet(pkl_path, iterable)
        else:
            return CachedSet(pkl_path)

    @_tag('supplement')
    def _supplement_support(self, db, start_date, end_date):
        """Calculate the support for the given date range of pa statements."""
        # If we are continuing, check for support links that were already found
        new_support_links = self._get_cached_set('new_support_links')
        npa_done = self._get_cached_set('npa_done')

        self._log("Downloading all pre-existing support links")
        existing_links = {(a, b) for a, b in
                          db.select_all([db.PASupportLinks.supported_mk_hash,
                                         db.PASupportLinks.supporting_mk_hash])}
        # Just in case...
        new_support_links -= existing_links

        # Now find the new support links that need to be added.
        batching_args = (self.batch_size,
                         db.PAStatements.json,
                         db.PAStatements.create_date >= start_date,
                         db.PAStatements.create_date <= end_date)
        npa_json_iter = db.select_all_batched(*batching_args,
                                              order_by=db.PAStatements.mk_hash)
        for outer_idx, npa_json_batch in npa_json_iter:
            # Create the statements from the jsons.
            npa_batch = []
            for s_json, in npa_json_batch:
                s = _stmt_from_json(s_json)
                if s.get_hash(shallow=True) not in npa_done:
                    npa_batch.append(s)

            # Compare internally
            self._log("Getting support for new pa batch %d." % outer_idx)
            some_support_links = self._get_support_links(npa_batch)

            try:
                # Compare against the other new batch statements.
                other_npa_json_iter = db.select_all_batched(
                    *batching_args,
                    order_by=db.PAStatements.mk_hash,
                    skip_idx=outer_idx
                )
                for inner_idx, other_npa_json_batch in other_npa_json_iter:
                    other_npa_batch = [_stmt_from_json(s_json)
                                       for s_json, in other_npa_json_batch]
                    split_idx = len(npa_batch)
                    full_list = npa_batch + other_npa_batch
                    self._log("Comparing outer batch %d to inner batch %d of "
                              "other new statements." % (outer_idx, inner_idx))
                    some_support_links |= \
                        self._get_support_links(full_list, split_idx=split_idx)

                # Compare against the existing statements.
                opa_json_iter = db.select_all_batched(
                    self.batch_size,
                    db.PAStatements.json,
                    db.PAStatements.create_date < start_date
                )
                for opa_idx, opa_json_batch in opa_json_iter:
                    opa_batch = [_stmt_from_json(s_json)
                                 for s_json, in opa_json_batch]
                    split_idx = len(npa_batch)
                    full_list = npa_batch + opa_batch
                    self._log("Comparing new batch %d to batch %d of old "
                              "statements." % (outer_idx, opa_idx))
                    some_support_links |= \
                        self._get_support_links(full_list, split_idx=split_idx)
            finally:
                # Stash the new support links in case we crash.
                new_support_links |= (some_support_links - existing_links)
                new_support_links.dump()
                npa_done.dump()

            npa_done |= {s.get_hash(shallow=True) for s in npa_batch}

        # Insert any remaining support links.
        if new_support_links:
            self._log("Copying %d support links into db."
                      % len(new_support_links))
            db.copy('pa_support_links', new_support_links,
                    ('supported_mk_hash', 'supporting_mk_hash'))
        return

    @_preassembly_wrapper
    @_tag('supplement')
    def supplement_corpus(self, db, continuing=False):
        """Update the table of preassembled statements.

        This method will take any new raw statements that have not yet been
        incorporated into the preassembled table, and use them to augment the
        preassembled table.

        The resulting updated table is indistinguishable from the result you
        would achieve if you had simply re-run preassembly on _all_ the
        raw statements.
        """
        start_date, end_date = self._supplement_statements(db)
        self._supplement_support(db, start_date, end_date)
        return True

    def _log(self, msg, level='info'):
        """Applies a task specific tag to the log message."""
        if self.__print_logs:
            print("Preassembly Manager [%s] (%s): %s"
                  % (datetime.now(), self.__tag, msg))
        getattr(logger, level)("(%s) %s" % (self.__tag, msg))

    @clockit
    def _clean_statements(self):
        """Perform grounding, sequence mapping, and find unique set from stmts.

        This method returns a list of statement objects, as well as a set of
        tuples of the form (uuid, matches_key) which represent the links between
        raw (evidence) statements and their unique/preassembled counterparts.
        """
        self._log("Map grounding...")
        grounded_stmts = ac.map_grounding(self.stmts)
        self._log("Map sequences...")
        mapped_stmts = ac.map_sequence(grounded_stmts, use_cache=True)

        self.cleaned_stmts = mapped_stmts
        return

    @clockit
    def _get_support_links(self, unique_stmts, split_idx=None):
        """Find the links of refinement/support between statements."""
        id_maps = self.pa._generate_id_maps(unique_stmts, poolsize=self.n_proc,
                                            split_idx=split_idx)
        ret = set()
        for ix_pair in id_maps:
            if ix_pair[0] == ix_pair[1]:
                assert False, "Self-comparison occurred."
            hash_pair = \
                tuple([shash(unique_stmts[ix]) for ix in ix_pair])
            if hash_pair[0] == hash_pair[1]:
                assert False, "Input list included duplicates."
            ret.add(hash_pair)

        return ret


class CachedSet(set):

    def __init__(self, cache, *args, **kwargs):
        self.cache = cache
        super(CachedSet, self).__init__(*args, **kwargs)

    def dump(self):
        with open(self.cache, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, cache):
        if path.exists(cache):
            with open(cache, 'rb') as f:
                return pickle.load(f)
        return cls(cache)


def _stmt_from_json(stmt_json_bytes):
    return Statement._from_json(json.loads(stmt_json_bytes.decode('utf-8')))


# This is purely for reducing having to type this long thing so often.
def shash(s, refresh=False):
    """Get the shallow hash of a statement."""
    return s.get_hash(shallow=True, refresh=refresh)


def make_graph(unique_stmts, match_key_maps):
    """Create a networkx graph of the statement and their links."""
    import networkx as nx
    g = nx.Graph()
    link_matches = {m for l in match_key_maps for m in l}
    unique_stmts_dict = {}
    for stmt in unique_stmts:
        if stmt.matches_key() in link_matches:
            g.add_node(stmt)
            unique_stmts_dict[stmt.matches_key()] = stmt

    for k1, k2 in match_key_maps:
        g.add_edge(unique_stmts_dict[k1], unique_stmts_dict[k2])

    return g


def _make_parser():
    parser = ArgumentParser(
        description='Manage preassembly of raw statements into pa statements.'
    )
    parser.add_argument(
        choices=['create', 'update'],
        dest='task',
        help=('Choose whether you want to perform an initial upload or update '
              'the existing content on the database.')
    )
    parser.add_argument(
        '-c', '--continue',
        dest='continuing',
        action='store_true',
        help='Continue uploading or updating, picking up where you left off.'
    )
    parser.add_argument(
        '-n', '--num_procs',
        dest='num_procs',
        type=int,
        default=None,
        help=('Select the number of processors to use during this operation. '
              'Default is 1.')
    )
    parser.add_argument(
        '-b', '--batch',
        type=int,
        default=10000,
        help=("Select the number of statements loaded at a time. More "
              "statements at a time will run faster, but require more memory.")
    )
    parser.add_argument(
        '-d', '--debug',
        dest='debug',
        action='store_true',
        help='Run with debugging level output.'
    )
    parser.add_argument(
        '-D', '--database',
        default='primary',
        help=('Choose a database from the names given in the config or '
              'environment, for example primary is INDRA_DB_PRIMAY in the '
              'config file and INDRADBPRIMARY in the environment. The default '
              'is \'primary\'.')
    )
    return parser


def _main():
    parser = _make_parser()
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        from indra.db.database_manager import logger as db_logger
        db_logger.setLevel(logging.DEBUG)
    print("Getting %s database." % args.database)
    db = get_db(args.database)
    assert db is not None
    db.grab_session()
    pm = PreassemblyManager(args.num_procs, args.batch)

    desc = 'Continuing' if args.continuing else 'Beginning'
    print("%s to %s preassembled corpus." % (desc, args.task))
    if args.task == 'create':
        pm.create_corpus(db, args.continuing)
    elif args.task == 'update':
        pm.supplement_corpus(db, args.continuing)
    else:
        raise IndraDBPreassemblyError('Unrecognized task: %s.' % args.task)


if __name__ == '__main__':
    _main()
