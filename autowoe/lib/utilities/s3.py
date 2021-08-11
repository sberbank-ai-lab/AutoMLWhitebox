"""S3 wrapper."""
# flake8: noqa

import s3fs


class S3Client(s3fs.S3FileSystem):
    """Класс-обёртка для доступа к хранилищу S3.

    Используется для доступа к объектам хранилища с использованием интерфейса файловой системы.

    """

    def __init__(self, aws_access_key_id, aws_secret_access_key, namespace=None, endpoint_url=None, **kwargs):
        """Конструктор объекта файловой системы на S3 SberCloud.

        Args:
            aws_access_key_id: Публичный ключ доступа к бакету S3
            aws_secret_access_key: Приватный ключ доступа к бакету S3
            namespace: Идентификатор пространства пользователя в хранилище SberCloud. Используется для формировании
                URL web-сервиса S3 SberCloud. Если не задан, то необходимо задать URL в аргументе endpoint_url
            endpoint_url: URL web-сервиса S3 SberCloud. Если не задан, то URL будет автоматически
                сконструирован на основании значения namespace.
            kwargs: Дополнительные параметры, передаваемые конструктору s3fs.S3FileSystem

        """
        if not namespace and not endpoint_url:
            raise ValueError("Either namespace or endpoint_url is required")

        self.namespace = namespace
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.endpoint_url = endpoint_url

        super(S3Client, self).__init__(
            key=self.aws_access_key_id,
            secret=self.aws_secret_access_key,
            client_kwargs={"endpoint_url": self.endpoint_url},
            **kwargs
        )
