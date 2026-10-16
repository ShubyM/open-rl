import { test } from 'node:test';
import assert from 'node:assert/strict';
import { samplerPodPattern } from '../../src/server/dashboard/ui/format.js';

test('shared, dedicated, historical and pending samplers use exact pod identities', () => {
  const run = { run_id: 'logical-run', pods: [{role: 'sampler', name: 'orw-lora-qwen-0-sampler'}, {role: 'trainer', name: 'trainer'}],
    workloads: [{role: 'sampler', pod_name: 'replica-1'}, {role: 'sampler', pod_name: null}] };
  const past = [{role:'sampler', run_ids:['logical-run'], pod:'old-pod'}, {role:'sampler', run_ids:['peer'], pod:'other-pod'}];
  const regex = new RegExp(samplerPodPattern(run, past));
  for (const name of ['orw-lora-qwen-0-sampler', 'replica-1', 'old-pod']) assert.ok(regex.test(name));
  for (const name of ['other-pod', 'trainer', 'prefix-replica-1', 'replica-1-suffix', '']) assert.ok(!regex.test(name));
  assert.equal(samplerPodPattern({run_id:'unplaced'}), null);
  assert.equal(samplerPodPattern(undefined), null);
  assert.equal(samplerPodPattern({pods:[{role:'sampler',name:23}]}), null);
});

test('fuzz pod filters: regex metacharacters and URL round trips never broaden scope', () => {
  let seed = 1739;
  const next = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0);
  const alphabet = 'abcXYZ09-_.+*?^$()[]{}|\\/&= #%é';
  for (let i = 0; i < 2000; i++) {
    const names = Array.from({length: 1 + next()%8}, () => Array.from({length:1+next()%40}, () => alphabet[next()%alphabet.length]).join(''));
    const pattern = samplerPodPattern({pods:names.map(name=>({role:'sampler',name}))});
    const params = new URLSearchParams({'var-pod':pattern});
    const regex = new RegExp(new URLSearchParams(params.toString()).get('var-pod'));
    for (const name of names) { assert.ok(regex.test(name)); assert.ok(!regex.test('prefix-'+name+'-suffix')); }
    assert.equal(pattern, samplerPodPattern({pods:[...names,...names].reverse().map(name=>({role:'sampler',name}))}));
  }
});
