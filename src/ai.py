from ai_bricks.api import openai
import stats
import os

DEFAULT_USER = os.getenv('COMMUNITY_USER','')

def use_key(key):
	openai.use_key(key)

usage_stats = stats.get_stats(user=DEFAULT_USER)
def set_user(user):
	global usage_stats
	usage_stats = stats.get_stats(user=user)
	openai.set_global('user', user)
	openai.add_callback('after', stats_callback)

def complete(text, **kw):
	model = kw.get('model','gpt-3.5-turbo')
	llm = openai.model(model)
	llm.config['pre_prompt'] = 'output only in raw text' # for chat models
	resp = llm.complete(text, **kw)
	resp['model'] = model
	return resp

def embedding(text, **kw):
	model = kw.get('model','text-embedding-ada-002')
	llm = openai.model(model)
	resp = llm.embed(text, **kw)
	resp['model'] = model
	return resp

def embeddings(texts, **kw):
	model = kw.get('model','text-embedding-ada-002')
	llm = openai.model(model)
	resp = llm.embed_many(texts, **kw)
	# Convert usage object to dictionary if needed
	if hasattr(resp['usage'], 'prompt_tokens'):
		resp['usage'] = {
			'prompt_tokens': resp['usage'].prompt_tokens,
			'completion_tokens': resp['usage'].completion_tokens,
			'total_tokens': resp['usage'].total_tokens
		}
	resp['model'] = model
	return resp

tokenizer_model = openai.model('text-davinci-003')
def get_token_count(text):
	return tokenizer_model.token_count(text)

def stats_callback(out, resp, self):
	model = self.config['model']
	# Handle both dictionary-style and attribute-style responses
	usage = resp.usage if hasattr(resp, 'usage') else resp['usage']

	# Convert usage to dictionary if it's an object
	if hasattr(usage, 'prompt_tokens'):
		usage = {
			'prompt_tokens': usage.prompt_tokens,
			'total_tokens': usage.total_tokens,
			'completion_tokens': getattr(usage, 'completion_tokens', 0)
		}

	usage['call_cnt'] = 1
	if 'text' in out:
		usage['completion_chars'] = len(out['text'])
	elif 'texts' in out:
		usage['completion_chars'] = sum([len(text) for text in out['texts']])
	# TODO: prompt_chars
	# TODO: total_chars
	if 'rtt' in out:
		usage['rtt'] = out['rtt']
		usage['rtt_cnt'] = 1

	usage_stats.incr(f'usage:v4:[date]:[user]', {f'{k}:{model}':v for k,v in usage.items()})
	usage_stats.incr(f'hourly:v4:[date]',       {f'{k}:{model}:[hour]':v for k,v in usage.items()})
	#print('STATS_CALLBACK', usage, flush=True) # XXX

def get_community_usage_cost():
	data = usage_stats.get(f'usage:v4:[date]:{DEFAULT_USER}')
	used = 0.0
	used += 0.04   * data.get('total_tokens:gpt-4',0) / 1000 # prompt_price=0.03 but output_price=0.06
	used += 0.02   * data.get('total_tokens:text-davinci-003',0) / 1000
	used += 0.002  * data.get('total_tokens:text-curie-001',0) / 1000
	used += 0.002  * data.get('total_tokens:gpt-3.5-turbo',0) / 1000
	used += 0.0004 * data.get('total_tokens:text-embedding-ada-002',0) / 1000
	return used
