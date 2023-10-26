import qrcode
from PIL import Image
from pyzbar.pyzbar import decode


class BasicQR:
    """
    Main class to read/generate basic (target) QR-codes that are readable by any device
    """

    @staticmethod
    def generate(text: str, box_size: int = 10, border: int = 4) -> Image:
        """
        Generates valid qr-code for a given `text`
            > QR version is determined automatically
            > Error correction is at minimal (for smaller image sizes)
            > Black/white colors is used
        """

        # define generator
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=box_size,
            border=border
        )

        # add text and determine qr version
        qr.add_data(text)
        qr.make(fit=True)  # automatically determine qr version

        # generate and convert to PIL.Image
        return qr.make_image().convert('RGB')

    @staticmethod
    def read(img: Image) -> str or int:
        """
        Reads text data from given QR
            > Works on zbar-tools [please download them using apt-get]
            > If several qr-codes found, return only one (this functionallity is enough for this task)
            > Reader is robust enough so that even stable-diffusion generated once works nicely
            > If no valid QR-codes found, return -1
        """

        # decode image
        decoded = decode(img)

        # if nothing found, print warning and return `-1`
        if not decoded:
            print('Failed to find any QR')
            return -1
        return decoded[0].data.decode("utf-8")
    