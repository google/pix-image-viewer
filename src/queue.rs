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

use crate::stats;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use std::boxed::FnBox;
use std::collections::HashSet;
use std::time::Instant;

type WorkFn<T> = Box<FnBox() -> T + Send + 'static>;

type Sammich<T> = Option<(usize, T, Instant)>;

type In<T> = Sammich<WorkFn<T>>;

type Out<T> = Sammich<T>;

pub struct Queue<T> {
    tx: Sender<In<T>>,
    rx: Receiver<Out<T>>,
    inflight: HashSet<usize>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl<T: 'static + Send> Queue<T> {
    pub fn new(threads: u32) -> Self {
        let (in_tx, in_rx) = bounded(1);
        let (out_tx, out_rx) = unbounded();

        let mut handles = Vec::new();
        for _ in 0..threads {
            let (in_rx, out_tx) = (in_rx.clone(), out_tx.clone());
            handles.push(std::thread::spawn(move || Self::worker(in_rx, out_tx)));
        }

        Self {
            tx: in_tx,
            rx: out_rx,
            inflight: HashSet::new(),
            handles,
        }
    }

    fn worker(rx: Receiver<In<T>>, tx: Sender<Out<T>>) {
        while let Some((i, work_fn, e2e)) = rx.recv().expect("worker recv") {
            let result = work_fn();
            tx.send(Some((i, result, e2e))).expect("worker send");
        }
        tx.send(None).expect("worker shutdown send");
    }

    pub fn send(&mut self, i: usize, work_fn: WorkFn<T>) {
        assert!(self.inflight.insert(i));

        self.tx
            .send(Some((i, work_fn, Instant::now())))
            .expect("queue send");
    }

    pub fn recv(&mut self) -> Option<(usize, T)> {
        if let Ok(Some((i, result, e2e))) = self.rx.try_recv() {
            assert!(self.inflight.remove(&i));

            stats::record("queue_e2e", e2e.elapsed());

            Some((i, result))
        } else {
            None
        }
    }

    pub fn inflight(&self, i: usize) -> bool {
        self.inflight.contains(&i)
    }

    pub fn is_full(&mut self) -> bool {
        self.tx.is_full()
    }
}

impl<T> Drop for Queue<T> {
    fn drop(&mut self) {
        info!("Sending worker terminations");
        for _ in 0..self.handles.len() {
            self.tx.send(None).expect("drop send");
        }
        info!("Waiting for worker shutdown");
        for _ in 0..self.handles.len() {
            while let Some(_) = self.rx.recv().expect("drop recv") {}
        }
    }
}
