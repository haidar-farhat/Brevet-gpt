"""Add a MariaDB FULLTEXT index on chunk content for lexical (BM25-style)
search, complementing the dense vectors in Chroma to enable hybrid retrieval.

Django has no built-in FULLTEXT index for MySQL/MariaDB, so this is raw SQL.
The accent-insensitive utf8mb4_unicode_ci collation means a search for
'eleve' also matches 'élève', which is what we want for French.
"""
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("catalog", "0001_initial"),
    ]

    operations = [
        migrations.RunSQL(
            sql="ALTER TABLE chunks ADD FULLTEXT INDEX ft_chunk_content (content);",
            reverse_sql="ALTER TABLE chunks DROP INDEX ft_chunk_content;",
        ),
    ]
