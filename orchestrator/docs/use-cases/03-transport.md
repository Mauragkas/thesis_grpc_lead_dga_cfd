# Use Case: Build Transport Endpoint and Wait for Readiness

## Description

Builds a tonic endpoint from a target string and waits until the channel is ready before making gRPC calls.

## Primary actor

- Orchestrator

## Supporting systems

- tonic transport
- Evaluator service or LEAD node

## Main steps: build endpoint

1. Receive a target string.
2. If it lacks `http://` or `https://`, prefix it with `http://`.
3. Apply transport settings:
   - keep-alive while idle
   - keep-alive timeout
   - TCP keepalive
   - connect timeout
   - request timeout
4. Return the configured `Endpoint`.

## Main steps: wait for channel

1. Compute a deadline from the current time plus the configured timeout.
2. Attempt to connect to the endpoint.
3. If the connection succeeds, return the channel.
4. If it fails, sleep briefly and retry until the deadline.
5. If the deadline is reached, return `Status::unavailable`.

## Postconditions

- A live `Channel` is available for downstream gRPC clients.

## Failure cases

- Invalid URI
- Endpoint unreachable
- Deadline exceeded
