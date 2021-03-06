---
title: Cookies are passé; browser fingerprinting is à la mode
layout: blog_outline
---

You probably know browser cookies as the reason why you’ll Google something and then suddenly start seeing advertisements for it on your Instagram feed. More specifically, this spooky action is the work of third-party cookies – little bits of data stored in your browser by advertising companies, in order to collect data on your browsing habits.

Since the start of the web, online advertising companies have figured out that the most efficient way to do online advertising is to show you advertisements that line up with your interests. Third party cookies allow these advertising companies to track your movements across the web, and glean valuable data about your life.

If this feels like a massive invasion of privacy to you, you’re not alone. At the beginning of this year,[ Google announced](https://blog.chromium.org/2020/01/building-more-private-web-path-towards.html) that it would end support for third-party tracking cookies before 2022 – essentially a death sentence for this longtime staple of online advertising and tracking, considering [close to 70%](https://gs.statcounter.com/browser-market-share) of all internet users worldwide use Google Chrome. The Safari and Firefox browsers have already begun blocking tracking cookies automatically.

But it’s going to take a lot more for advertising companies to throw in their towels and give up on tracking your browsing. Blocking third party cookies is a good first step, but advertising companies are much more clever than we think – after all, their whole livelihood depends on collecting your data.
Enter browser fingerprinting: a seemingly innocuous scheme that has big implications for online privacy in a post-cookie era. Browser fingerprinting collects a number of small, seemingly useless pieces of data about your browser, including your default fonts, page zoom levels, the size of your screen, time zone, and over 20 other unique identifiers of the hardware and software on your computer. The sum of these properties is called your browser fingerprint. These are fairly common properties – after all, there are surely a lot of people in your time zone. It’s quite likely that you and someone you know share the same one crease on your thumb – but the likelihood of your entire fingerprints matching is incredibly small. 

Altogether, the likelihood of an exact browser fingerprint match for two people is incredibly small. 
When someone with the same fingerprint visits two different web pages, those two websites can be fairly confident that this is the same person. When you then log in to Facebook, Facebook (or any other site that verifies your identity) now knows that this fingerprint belongs to you – and it knows all the sites your fingerprint has appeared on. It’s game over for your privacy, once again. But this time, you can’t just disable cookies or install an adblocker to keep you safe.

You can check if your browser fingerprint is unique at [Panopticlick](https://panopticlick.eff.org), a project by the Electronic Frontier Foundation. It found my Safari browser to have a unique fingerprint among at least 228,159 other browsers. This is pretty bad.

Browser fingerprinting is a very powerful tactic and one that is extremely difficult to counter. One way is to try to make your fingerprint as common as possible – perhaps by running a standard version of a popular browser (Chrome, Firefox) on a popular operating system (Windows). This way advertisers will have a hard time telling you apart from the thousands of other people with the exact same fingerprint – but for most, changing computers is unfeasible.

One could also use the Tor browser or disable JavaScript, which is needed for most fingerprinting attempts – at the expense of having a pretty terrible browsing experience, since JavaScript is required for most websites to function.

The most reliable method to thwart fingerprinting is to constantly change your fingerprint, making it unique for every single site you visit. If advertising companies think that you are a new person on every site you go to, they can’t track you. I use the [Brave](https://brave.com/) browser, which generates random data instead of sending actual fingerprintable information about the device. This guarantees my fingerprint will be different on every site I visit, so advertisers can’t track the same fingerprint across multiple pages. 
In many ways, browser fingerprinting is scarier than third-party cookie tracking. It gives advertising companies great power over our online experiences, and without an easy way to regain control of our online identities. Being constantly vigilant of privacy threats like fingerprinting is the only way to make the internet more safe and secure.
