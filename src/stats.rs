// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TODO: Should be able to record event names and start/end times for visualization in a timeline.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

lazy_static! {
    static ref STATS: Mutex<BTreeMap<&'static str, histogram::Histogram>> =
        { Mutex::new(BTreeMap::new()) };
}

pub fn record(name: &str, duration: Duration) {
    let micros = duration.as_micros() as u64;

    let mut handle = STATS.lock().unwrap();

    if let Some(hist) = handle.get_mut(name) {
        hist.increment(micros).expect("increment");
    } else {
        let name = name.to_string().into_boxed_str();
        let name: &'static str = Box::leak(name);

        let mut hist = histogram::Config::new()
            .precision(3)
            .build()
            .expect("build hist");
        hist.increment(micros).expect("increment");

        handle.insert(name, hist);
    }
}

pub fn dump() {
    let handle = STATS.lock().unwrap();

    let uptime = handle["uptime"].mean().expect("uptime") as f64;

    println!(
        "{name:<20} {count:>10} {p:>40} {total:>10} {pct:>10}",
        name = "name",
        count = "count",
        p = "p50/p90/p99/p100",
        total = "total(ms)",
        pct = "total(%)"
    );

    for (name, hist) in &*handle {
        if name == &"uptime" {
            continue;
        }

        let count = hist.entries();
        let mean = hist.mean().unwrap_or(0);

        let p = format!(
            "{p50}/{p90:}/{p99:}/{p100:}",
            p50 = hist.percentile(50.0).expect("p50"),
            p90 = hist.percentile(90.0).expect("p90"),
            p99 = hist.percentile(99.0).expect("p99"),
            p100 = hist.percentile(100.0).expect("p100"),
        );

        println!(
            "{name:<20} {count:>10} {p:>40} {total:>10} {pct:>10.2}",
            name = name,
            count = count,
            p = p,
            total = (mean * count) / 1_000,
            pct = ((mean * count) as f64) / uptime * 100.0,
        );
    }
}

#[derive(Debug)]
pub struct ScopedDuration<'a> {
    name: &'a str,
    instant: Instant,
}

impl<'a> ScopedDuration<'a> {
    pub fn new(name: &'a str) -> Self {
        Self {
            name,
            instant: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.instant.elapsed()
    }
}

impl<'a> Drop for ScopedDuration<'a> {
    fn drop(&mut self) {
        record(self.name, self.instant.elapsed());
    }
}
