"""Read owner-scoped GPU quota and every latest GPU batch version; never change credentials.

Run through kaggle_as.py <account> -- python scripts/kaggle_inventory.py <account>.
Unknown status refuses allocation instead of treating a query failure as a free slot.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetAcceleratorQuotaStatisticsRequest, ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiListKernelsRequest,
)
from kagglesdk.kernels.types.kernels_enums import KernelsListSortType, KernelsListViewType
from kagglesdk.security.types.oauth_service import IntrospectTokenRequest


def inventory(account):
    token = os.environ['KAGGLE_API_TOKEN']
    if token.startswith('/'):
        token = Path(token).read_text().strip().removeprefix('Bearer ').strip()
    api = KaggleApi.build_kaggle_client_with_params(args=[], api_token=token)
    with api:
        identity = IntrospectTokenRequest()
        identity.token = token
        who = api.security.oauth_client.introspect_token(identity)
        assert who.active and who.username == account, 'account identity mismatch'
        quota = api.kernels.kernels_api_client.get_accelerator_quota_statistics(
            ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        left = max(0, (quota.total_time_allowed - quota.time_used - quota.time_reserved).total_seconds() / 3600)
        request = ApiListKernelsRequest()
        request.group, request.user = KernelsListViewType.PROFILE, account
        request.sort_by, request.page_size = KernelsListSortType.DATE_RUN, 100
        kernels = []
        while True:
            page = api.kernels.kernels_api_client.list_kernels(request)
            kernels.extend(page.kernels)
            if not page.next_page_token:
                break
            request.page_token = page.next_page_token
        def inspect(kernel):
            slug = kernel.ref.split('/')[-1]  # list API leaves slug/accelerator/version fields at defaults
            version, active = kernel.current_version_number or None, None
            status = ApiGetKernelSessionStatusRequest()
            status.user_name, status.kernel_slug = account, slug
            result = api.kernels.kernels_api_client.get_kernel_session_status(status).status.name
            if result not in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                request = ApiGetKernelRequest()
                request.user_name, request.kernel_slug = account, slug
                metadata = api.kernels.kernels_api_client.get_kernel(request).metadata
                version = metadata.current_version_number or None
                if metadata.enable_gpu:
                    active = {'ref': kernel.ref, 'version': version, 'status': result}
            return slug, version, active
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked = list(pool.map(inspect, kernels))
        busy = [active for _, _, active in checked if active]
        versions = {slug: version for slug, version, _ in checked}
    return {'account': account, 'gpu_left': left, 'busy': busy, 'versions': versions}


if __name__ == '__main__':
    try:
        print(json.dumps(inventory(sys.argv[1])))
    except Exception as exc:
        sys.exit(f'inventory unavailable: {type(exc).__name__}; credential details suppressed')
