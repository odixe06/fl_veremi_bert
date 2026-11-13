"""The live list API includes empty placeholders and leaves accelerator fields at defaults."""
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from scripts import kaggle_inventory as module


@pytest.mark.parametrize('query_fails', [False, True])
def test_inventory_checks_real_refs_and_preserves_query_errors(monkeypatch, query_fails):
    seen = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def introspect_token(self, request):
            return NS(active=True, username='owner')

        def get_accelerator_quota_statistics(self, request):
            return NS(gpu_quota=NS(total_time_allowed=timedelta(hours=30), time_used=timedelta(), time_reserved=timedelta()))

        def list_kernels(self, request):
            return NS(kernels=[NS(ref='', current_version_number=0),
                               NS(ref='owner/done', current_version_number=0, enable_gpu=False)], next_page_token='')

        def get_kernel_session_status(self, request):
            assert request.user_name == 'owner' and request.kernel_slug == 'done'
            seen.append(request.kernel_slug)
            if query_fails:
                raise TimeoutError('status unknown')
            return NS(status=NS(name='COMPLETE'))

    client = Client()
    client.security = NS(oauth_client=client)
    client.kernels = NS(kernels_api_client=client)
    monkeypatch.setenv('KAGGLE_API_TOKEN', 'fake-token-for-test')
    monkeypatch.setattr(module.KaggleApi, 'build_kaggle_client_with_params', lambda **kw: client)
    if query_fails:
        with pytest.raises(TimeoutError):
            module.inventory('owner')
    else:
        result = module.inventory('owner')
        assert result['gpu_left'] == 30 and result['busy'] == [] and result['versions'] == {'done': None}
    assert seen == ['done']
