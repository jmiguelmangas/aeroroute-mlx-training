# Review protocol for ranked synthetic examples

Each queued example must be approved by a human reviewer before it is used as a promotion corpus.

- Check that the stated winner is the candidate with the lowest supplied fuel.
- Check that every number in the explanation exists in the supplied alternatives.
- Check that the wording is explanatory and contains no operational flight advice.
- Record the reviewer, date, and decision outside Git with the dataset manifest checksum.

Until that review is complete, this corpus is training material only and cannot promote an adapter.
