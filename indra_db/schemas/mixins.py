import logging
from psycopg2.errors import DuplicateTable
from sqlalchemy import inspect, Column, BigInteger
from sqlalchemy.exc import NoSuchTableError

logger = logging.getLogger(__name__)


class IndraDBTable(object):
    _indices = []
    _skip_disp = []
    _always_disp = ['id']

    @classmethod
    def create_index(cls, db, index, commit=True):
        inp_data = {'idx_name': index.name,
                    'full_name': cls.full_name(force_schema=True),
                    'idx_def': index.definition}
        sql = ("CREATE INDEX {idx_name} ON {full_name} "
               "USING {idx_def} TABLESPACE pg_default;".format(**inp_data))
        if commit:
            try:
                cls.execute(db, sql)
            except DuplicateTable as err:
                logger.warning("Got error (%s) when building %s. Skipping."
                               % (err, index.name))
        return sql

    @classmethod
    def build_indices(cls, db):
        for index in cls._indices:
            print("Building index: %s" % index.name)
            cls.create_index(db, index)

    @classmethod
    def full_name(cls, force_schema=False):
        """Get the full name including the schema, if supplied."""
        name = cls.__tablename__

        # If we are definitely going to include the schema, default to public
        if force_schema:
            schema_name = 'public'
        else:
            schema_name = None

        # Look for any information in the __table_args__ about the schema
        if isinstance(cls.__table_args__, dict):
            schema_name = cls.__table_args__.get('schema')
        elif isinstance(cls.__table_args__, tuple):
            for arg in cls.__table_args__:
                if isinstance(arg, dict) and 'schema' in arg.keys():
                    schema_name = arg['schema']
                    break

        # Prepend if we found something.
        if schema_name:
            name = schema_name + '.' + name

        return name

    def _make_str(self):
        s = self.full_name() + ':\n'
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if k in self._skip_disp:
                    s += '\t%s: [not shown]\n' % k
                else:
                    s += '\t%s: %s\n' % (k, v)
        return s

    def display(self):
        """Display the values of this entry."""
        print(self._make_str())

    def __str__(self):
        return self._make_str()

    def __repr__(self):
        ret = self.__class__.__name__ + '('

        entries = []
        for attr_name in self._always_disp:
            attr = getattr(self, attr_name)
            if isinstance(attr, str):
                fmt = '%s="%s"'
            elif isinstance(attr, bytes):
                fmt = '%s=b"%s"'
            else:
                fmt = '%s=%s'

            entries.append(fmt % (attr_name, attr))

        ret += ', '.join(entries)
        ret += ')'
        return ret


class ReadonlyTable(IndraDBTable):
    __definition__ = NotImplemented
    __table_args__ = NotImplemented

    @classmethod
    def create(cls, db, commit=True):
        sql = "CREATE TABLE IF NOT EXISTS %s.%s AS %s;" \
              % (cls.__table_args__['schema'], cls.__tablename__,
                 cls.get_definition())
        if commit:
            cls.execute(db, sql)
        return sql

    @classmethod
    def get_definition(cls):
        return cls.__definition__

    @staticmethod
    def execute(db, sql):
        conn = db.engine.raw_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return


class SpecialColumnTable(ReadonlyTable):

    @classmethod
    def create(cls, db, commit=True):

        # Create the materialized view.
        cls.__definition__ = cls.definition(db)
        sql = super(cls, cls).create(db, commit)
        cls.loaded = True
        return sql

    @classmethod
    def load_cols(cls, engine):
        if cls.loaded:
            return

        try:
            schema = cls.__table_args__.get('schema', 'public')
            cols = inspect(engine).get_columns(cls.__tablename__,
                                               schema=schema)
        except NoSuchTableError:
            return

        existing_cols = {col.name for col in cls.__table__.columns}
        for col in cols:
            if col['name'] in existing_cols:
                continue

            setattr(cls, col['name'], Column(BigInteger))

        cls.loaded = True
        return


class NamespaceLookup(ReadonlyTable):
    __dbname__ = NotImplemented

    @classmethod
    def get_definition(cls):
        return ("SELECT db_id, ag_id, role, ag_num, type, "
                "mk_hash, ev_count FROM readonly.pa_meta "
                "WHERE db_name = '%s'" % cls.__dbname__)

