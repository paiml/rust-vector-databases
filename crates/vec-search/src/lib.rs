//! Hybrid-search helpers: reciprocal-rank fusion (RRF) plus a
//! tiny filter-builder for payload conditions.
//!
//! Both modules are pure data-structure code — no IO, no async, no
//! Qdrant types. The Qdrant adapter (`vec-qdrant`) and the in-process
//! adapter (`vec-aprender`) consume `FilteredQuery` and translate it
//! to whatever filter shape their backend expects.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]

use std::collections::HashMap;

/// Reciprocal Rank Fusion. Combines two ranked id lists into a single
/// ranking by giving each id a score of `Σ 1/(k + rank_i)` across the
/// lists where it appears.
///
/// `k` is the conventional RRF smoothing constant — `60` is the
/// industry default (Cormack 2009). The fused output is sorted
/// descending by score; ties keep the first-appearance order.
///
/// # Example
///
/// ```
/// use vec_search::RrfFusion;
/// let dense  = vec![10, 20, 30];
/// let sparse = vec![20, 10, 40];
/// let fused = RrfFusion::new(60.0).fuse(&[&dense, &sparse]);
/// // 10 ranks first in dense + second in sparse
/// // 20 ranks second in dense + first in sparse
/// // both share the top of the fused ranking; 10 wins by tie-break order
/// assert_eq!(fused[0].0, 10);
/// assert_eq!(fused[1].0, 20);
/// ```
pub struct RrfFusion {
    k: f32,
}

impl RrfFusion {
    /// Construct an RRF fuser with the given smoothing constant `k`.
    /// `60.0` is the literature-standard default — keep it unless you
    /// have a calibration reason to change.
    #[must_use]
    pub fn new(k: f32) -> Self {
        Self { k }
    }

    /// Default RRF smoothing constant (`k = 60.0`).
    #[must_use]
    pub fn default_k() -> Self {
        Self::new(60.0)
    }

    /// Fuse a list of ranked id arrays into one ranking.
    ///
    /// Returns a `Vec<(id, score)>` sorted by score descending.
    /// Ids that appear in none of the inputs are dropped.
    #[must_use]
    pub fn fuse<I: Copy + std::hash::Hash + Eq>(&self, lists: &[&[I]]) -> Vec<(I, f32)> {
        let mut scores: HashMap<I, f32> = HashMap::new();
        let mut first_seen: HashMap<I, usize> = HashMap::new();
        let mut order = 0usize;

        for list in lists {
            for (rank, &id) in list.iter().enumerate() {
                #[allow(clippy::cast_precision_loss)]
                let contribution = 1.0_f32 / (self.k + (rank + 1) as f32);
                *scores.entry(id).or_insert(0.0) += contribution;
                first_seen.entry(id).or_insert_with(|| {
                    let v = order;
                    order += 1;
                    v
                });
            }
        }

        let mut fused: Vec<(I, f32)> = scores.into_iter().collect();
        // Sort descending by score; tie-break by first-seen insertion order
        // so the result is deterministic.
        fused.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| first_seen[&a.0].cmp(&first_seen[&b.0]))
        });
        fused
    }
}

impl Default for RrfFusion {
    fn default() -> Self {
        Self::default_k()
    }
}

/// One condition on a payload field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldCondition {
    /// Field must exactly match the given string.
    Eq(String, String),
    /// Field must be one of the given strings.
    In(String, Vec<String>),
    /// Field must NOT match the given string.
    NotEq(String, String),
}

impl FieldCondition {
    /// Build an equality condition (`field == value`).
    #[must_use]
    pub fn eq(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Eq(field.into(), value.into())
    }

    /// Build a set-membership condition (`field in {values}`).
    #[must_use]
    pub fn one_of(field: impl Into<String>, values: Vec<String>) -> Self {
        Self::In(field.into(), values)
    }

    /// Build a negation condition (`field != value`).
    #[must_use]
    pub fn not_eq(field: impl Into<String>, value: impl Into<String>) -> Self {
        Self::NotEq(field.into(), value.into())
    }
}

/// Filter-then-search query builder. Backend-agnostic — emits a
/// vector of [`FieldCondition`]s the adapter translates to its
/// native filter syntax.
#[derive(Debug, Clone, Default)]
pub struct FilteredQuery {
    must: Vec<FieldCondition>,
    must_not: Vec<FieldCondition>,
}

impl FilteredQuery {
    /// New empty query — every condition you add tightens the filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a positive condition (every result must satisfy it).
    #[must_use]
    pub fn must(mut self, cond: FieldCondition) -> Self {
        self.must.push(cond);
        self
    }

    /// Add a negative condition (results matching it are excluded).
    #[must_use]
    pub fn must_not(mut self, cond: FieldCondition) -> Self {
        self.must_not.push(cond);
        self
    }

    /// Read the positive conditions.
    #[must_use]
    pub fn must_conditions(&self) -> &[FieldCondition] {
        &self.must
    }

    /// Read the negative conditions.
    #[must_use]
    pub fn must_not_conditions(&self) -> &[FieldCondition] {
        &self.must_not
    }

    /// True when no conditions have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.must.is_empty() && self.must_not.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_combines_two_lists() {
        let dense = vec![1, 2, 3];
        let sparse = vec![2, 1, 4];
        let fused = RrfFusion::new(60.0).fuse(&[&dense, &sparse]);
        // Both 1 and 2 appear in both lists at adjacent ranks; 1 has rank
        // (0, 1) and 2 has rank (1, 0); 4 only in sparse, 3 only in dense.
        let ids: Vec<i32> = fused.iter().map(|(id, _)| *id).collect();
        assert!(ids.starts_with(&[1, 2]) || ids.starts_with(&[2, 1]));
        assert!(ids.contains(&3));
        assert!(ids.contains(&4));
    }

    #[test]
    fn rrf_one_list_is_identity_ranking() {
        let only = vec![10, 20, 30];
        let fused = RrfFusion::default().fuse(&[&only]);
        let ids: Vec<i32> = fused.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn rrf_higher_rank_means_higher_score() {
        let only = vec![5, 6];
        let fused = RrfFusion::new(60.0).fuse(&[&only]);
        // rank 1 (id 5) > rank 2 (id 6) by RRF formula 1/(60+1) > 1/(60+2)
        assert!(fused[0].1 > fused[1].1);
    }

    #[test]
    fn rrf_empty_input_is_empty_output() {
        let empty: Vec<i32> = vec![];
        let fused = RrfFusion::default().fuse::<i32>(&[&empty]);
        assert!(fused.is_empty());
    }

    #[test]
    fn rrf_no_lists_is_empty() {
        let fused = RrfFusion::default().fuse::<i32>(&[]);
        assert!(fused.is_empty());
    }

    #[test]
    fn filtered_query_starts_empty() {
        let q = FilteredQuery::new();
        assert!(q.is_empty());
        assert!(q.must_conditions().is_empty());
        assert!(q.must_not_conditions().is_empty());
    }

    #[test]
    fn filtered_query_must_then_must_not() {
        let q = FilteredQuery::new()
            .must(FieldCondition::eq("rating", "PG"))
            .must(FieldCondition::one_of(
                "category",
                vec!["Action".to_string(), "Drama".to_string()],
            ))
            .must_not(FieldCondition::eq("language", "Klingon"));
        assert!(!q.is_empty());
        assert_eq!(q.must_conditions().len(), 2);
        assert_eq!(q.must_not_conditions().len(), 1);
        assert_eq!(
            q.must_conditions()[0],
            FieldCondition::Eq("rating".to_string(), "PG".to_string())
        );
        // must_not() keeps the inner FieldCondition as-is; the bucket
        // (must vs must_not) is what flips the polarity.
        assert_eq!(
            q.must_not_conditions()[0],
            FieldCondition::Eq("language".to_string(), "Klingon".to_string())
        );
    }

    #[test]
    fn field_condition_constructors_round_trip() {
        let a = FieldCondition::not_eq("year", "2026");
        assert_eq!(
            a,
            FieldCondition::NotEq("year".to_string(), "2026".to_string())
        );
    }
}
