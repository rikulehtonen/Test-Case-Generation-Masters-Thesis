from Browser import Browser
from Browser.utils.data_types import SupportedBrowsers
import time
import os
from Browser import AssertionOperator
import multiprocessing
import time

b = Browser(timeout="3000 s", retry_assertions_for="3000 ms", strict=False)
b.new_browser(headless=False, browser=SupportedBrowsers.chromium)
b.new_context(
    acceptDownloads=True,
    viewport={"width": 700, "height": 500}
)
b.new_page('localhost:3000')

b.click("xpath=//A[contains(text(),'Shop')]")
b.click("text='Add to Cart'")
b.click("xpath=//A[contains(text(),'Cart')]")
b.type_text("xpath=//INPUT[@type='text'][@name='name'][@placeholder='Full Name']", "test")
b.type_text("xpath=//INPUT[@type='email'][@name='email'][@placeholder='Email']", "test@test.fi")
b.type_text("xpath=//INPUT[@type='text'][@name='address'][@placeholder='Address']", "test")
b.click("xpath=//BUTTON[@type='submit'][contains(text(),'Submit Order')]")

time.sleep(3)

b.close_browser()