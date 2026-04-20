# Developer shortcuts (optional; CI uses uv directly).
# Fixed --man-date keeps committed man/stx.1 stable across regenerations.

MAN_DATE ?= 2026-04-19

.PHONY: man man-check

man:
	@mkdir -p man
	uv run click-man stx --target man \
		--man-version $$(grep '^version = ' pyproject.toml | sed 's/.*"\(.*\)".*/\1/') \
		--man-date $(MAN_DATE)

man-check: man
	@mkdir -p /tmp/stx-man-check
	uv run click-man stx --target /tmp/stx-man-check \
		--man-version $$(grep '^version = ' pyproject.toml | sed 's/.*"\(.*\)".*/\1/') \
		--man-date $(MAN_DATE)
	diff -rq man /tmp/stx-man-check
