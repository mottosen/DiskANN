/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann_vector::distance::{Distance, DistanceProvider};
use half::f16;

use crate::{
    counters::LocalCounters,
    epoch,
    num::Bytes,
    repr::{self, internal::Calf, quantization::Metric},
    store::{
        self,
        optional::Optional,
        simple::{self, Simple},
    },
};

/// Choose how data is going to be reranked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rerank {
    /// No reranking will be performed and no space for higher precision vectors will be
    /// allocated.
    None,

    /// Use 16-bit floating point numbers to store the higher precision representation.
    /// These will be used automatically during search to rerank candidates.
    F16,
}

/// Internal representation of [`Rerank`].
///
/// This is used for computing distances among the raw values in the auxiliary store.
#[derive(Debug)]
pub(in crate::repr) enum Reranker {
    None,
    F16(Distance<f32, f16>),
}

impl Reranker {
    /// Construct a new [`Reranker`] and a [`store::slots::SlotsConfig`]  for the auxiliary
    /// store.
    pub(in crate::repr) fn new_with_config(
        rerank: Rerank,
        metric: Metric,
        dim: usize,
    ) -> (Self, Option<simple::Config>) {
        let this = match rerank {
            Rerank::None => Self::None,
            Rerank::F16 => {
                let distance = <f32 as DistanceProvider<f16>>::distance_comparer(
                    metric.as_vector_metric(),
                    Some(dim),
                );

                Self::F16(distance)
            }
        };

        let config = match &this {
            Self::None => None,
            Self::F16(_) => Some(Simple::config(this.bytes_for(dim))),
        };

        (this, config)
    }

    #[expect(
        clippy::expect_used,
        reason = "the arithmetic should not overflow for the feasible `dim` values"
    )]
    fn bytes_for(&self, dim: usize) -> Bytes {
        match self {
            Self::None => Bytes::new(0),
            Self::F16(_) => Bytes::new(dim.checked_mul(2).expect("f16 is smaller than the f32")),
        }
    }

    /// Create a [`repr::PostProcess`].
    ///
    /// This assumes that `simple` has the same dimensions as `self`'s contained distance
    /// computation and that `guard` belongs to `simple`.
    ///
    /// # Pre-conditions
    ///
    /// This requires that `slots` is the [`store::slots::Slots`] created from the
    /// configuration returned in [`Self::new_with_config`].
    pub(in crate::repr) fn post_process<'a>(
        &'a self,
        query: &'a [f32],
        guard: &epoch::Guard<'a>,
        slots: &'a Optional<store::simple::Simple>,
        counters: &LocalCounters<'a>,
    ) -> Option<Box<dyn repr::PostProcess + 'a>> {
        match (self, slots.slots()) {
            (Self::None, None) => None,
            (Self::F16(distance), Some(simple)) => {
                let distance = repr::full::QueryDistance::new(Calf::Borrowed(query), *distance);
                let reader = simple.reader(guard.share());
                let post_process =
                    repr::internal::simple::Reranker::new(reader, distance, counters.fork());
                Some(Box::new(post_process))
            }
            _ => panic!("invalid combination of arguments"),
        }
    }

    /// Store the vector `v` into the raw buffer `buf`.
    ///
    /// # Pre-conditions
    ///
    /// `buf` must be consistent with the configuration returned from [`Self::new_with_config`],
    /// and may only be `None` if that configuration was `None`.
    ///
    /// If it is `Some`, this function may panic if its length is not consistent with the
    /// original configuration.
    pub(in crate::repr) fn store(&self, v: &[f32], buf: &mut Option<simple::Exclusive<'_>>) {
        match (self, buf) {
            (Self::None, None) => {}
            (Self::F16(_), Some(exclusive)) => {
                use diskann_vector::conversion::CastFromSlice;
                bytemuck::cast_slice_mut::<u8, f16>(exclusive.as_mut_slice()).cast_from_slice(v);
            }
            _ => panic!("invalid combination of arguments"),
        }
    }
}

/// Test that [`repr::PostProcess`] reranks correctly.
///
/// Pass all `ids` to [`repr::PostProcess::post_process`]. Verify that all ids not
/// present in `distances` have been removed and the remaining ids are present, sorted,
/// and have distance values matching those in `distances`.
#[cfg(test)]
pub(in crate::repr) fn test_rerank(
    post_process: &mut dyn repr::PostProcess,
    distances: hashbrown::HashMap<crate::num::SlotId, f32>,
    ids: &[crate::num::SlotId],
    ctx: &dyn std::fmt::Display,
) {
    use diskann::neighbor::Neighbor;

    use crate::num::SlotId;

    let mut buffer: Vec<_> = ids
        .iter()
        .map(|slot_id| Neighbor::new(slot_id.value(), 0.0))
        .collect();

    post_process.post_process(&mut buffer).unwrap();
    let mut previous = f32::NEG_INFINITY;
    assert_eq!(buffer.len(), distances.len(), "{ctx}");
    for neighbor in buffer.iter() {
        let current = *neighbor.distance();
        assert_eq!(current, distances[&SlotId(*neighbor.id())], "{ctx}",);

        assert!(
            current >= previous,
            "distances is not monotonically increasing, previous = {}, current = {} -- {}",
            previous,
            current,
            ctx,
        );

        previous = current;
    }

    assert!(
        previous > f32::NEG_INFINITY,
        "previous = {} -- {}",
        previous,
        ctx
    );
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use std::assert_matches;

    use crate::{
        num::{Capacity, MaxDegree},
        store::Store,
    };

    /// Create a reranker of dim 2 for the given config and metric.
    fn make_store(rerank: Rerank, metric: Metric) -> (Reranker, Store<Optional<Simple>>) {
        let dim = 2;
        let (reranker, config) = Reranker::new_with_config(rerank, metric, dim);

        let store = Store::new(
            store::Layout::new(Capacity::new(10), MaxDegree::new(0), 1),
            store::Config::new(),
            config,
        )
        .unwrap();

        (reranker, store)
    }

    #[test]
    fn test_disabled_rerank() {
        let (reranker, store) = make_store(Rerank::None, Metric::SquaredL2);
        assert_matches!(reranker, Reranker::None);
        assert!(
            store.slots().slots().is_none(),
            "Rerank::None should create empty optional slots"
        );
    }

    #[test]
    fn test_reranker_f16() {
        for metric in Metric::all() {
            let (reranker, store) = make_store(Rerank::F16, metric);


        }
    }
}
